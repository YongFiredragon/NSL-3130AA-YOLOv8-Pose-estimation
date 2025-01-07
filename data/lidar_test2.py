import socket
import struct
import threading
import time
import numpy as np
import cv2


START_MARKER = b'\xff\xff\xaa\x55'
END_MARKER = b'\xff\xff\x55\xaa'


class ImageFrame:
    """ C++의 ImageFrame 구조체 유사 """
    def __init__(self):
        self.frameMat   = None  # amplitude or grayscale (cv::Mat -> np.ndarray)
        self.distMat    = None  # distance (cv::Mat -> np.ndarray)
        self.isRotate   = False
        self.pCatesianTable = None
        # 로컬 파일 테스트/이름 등은 생략


class NSL3130AA:
    """
    C++ NSL3130AA 클래스를 Python에서 객체지향(OOP) 형태로 옮긴 예시.
    - 멤버 변수: integrationTime3D, hdrMode, etc.
    - 멤버 함수: reqIntegrationTime(), reqHdrMode(), stopLidar(), etc.
    - sendToDev() (C++의 sendToDev) → Python에서는 self._send_packet() 등으로 구현
    """

    def __init__(self, ipaddr="192.168.0.220", port=50660):
        """
        C++ NSL3130AA 생성자와 비슷하게,
        기본값들 (integrationTime3D=800, hdrMode=0, etc.)을 세팅.
        """
        self.ipaddr = ipaddr
        self.port = port
        self.exit_thread    = False
        self.thread = None

        # 버퍼 (C++의 ring-buffer 구조를 간단화)
        self.buffer_lock = threading.Lock()
        self.frame_queue = []

        # 재조립용 변수들 초기화
        self.currentFrameNumber = -1
        self.completedFrame = None
        self.receivedBytes = 0

        # 소켓 관련
        self.control_sock = None

        # 색맵(거리/앰플리튜드용)
        # C++는 createColorMapPixel()을 통해 30000개 만든다. 여기선 간단히 OpenCV colormap 대체 가능
        self.color_map = cv2.COLORMAP_JET  # 간단 예시
        self.led_control = 1
        self.captureNetType = 0  # e.g. NONEMODEL_TYPE=0
        self.hdr_mode = 0  # e.g. DEFAULT_HDR_MODE=0 (HDR_NONE_MODE)
        self.integrationTime3D = 800
        self.integrationTime3DHdr1 = 100
        self.integrationTime3DHdr2 = 50
        self.integrationTimeGrayScale = 100

        self.hdr_mode = 0  # HDR_NONE
        self.led_control = 1
        self.minAmplitude = 50
        # 필요하면 ROI, overflow, compensation 등등 멤버도 추가
        self.rotate_90      = 0

        self.maxDist = 12500
        self.colorLUT = self.build_color_lut(30000)  # 한 번만 빌드

        print(f"[NSL3130AA] Created with ip={self.ipaddr}:{self.port}")

    def connect(self):
        """TCP(제어) + UDP(데이터) 소켓 연결"""
        # TCP (control)
        self.control_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.control_sock.connect((self.ipaddr, 50660))
        self.control_sock.settimeout(3.0)

        # UDP (data)
        self.data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.data_sock.bind(("", 45454))  # INADDR_ANY / port 45454
        self.data_sock.settimeout(3.0)

        # 초기화 커맨드 (C++: initializeTofcam660)
        self.initialize_device()

        # 스레드 시작
        self.exit_thread = False
        self.thread = threading.Thread(target=self.rx_loop, daemon=True)
        self.thread.start()

    def close(self):
        """C++ closeLidar() 일부(제어 소켓 정리)"""
        if self.control_sock:
            self.control_sock.close()
            self.control_sock = None
        print("[close] Socket closed.")

    def _send_packet(self, payload: bytes):
        """
        C++ sendToDev()에 해당.
        - start marker(4B)
        - length(4B, Big Endian) = len(payload)
        - payload
        - end marker(4B)
        """
        if not self.control_sock:
            raise ConnectionError("Not connected. Call connect() first.")

        length = len(payload)
        length_bytes = struct.pack('>I', length)  # Big-Endian 4B

        pkt = START_MARKER + length_bytes + payload + END_MARKER
        self.control_sock.sendall(pkt)

        # 여기서는 단순히 전송만, 응답을 받으려면 recv(...)
        print(f"[_send_packet] TX= {pkt.hex().upper()}")
        # 전송
        self.control_sock.sendall(pkt)
        print(f"[sendToDev] TX: {pkt.hex()}")

        # 응답 수신 (필요하면 timeout이나 예외처리 추가)
        try:
            resp = self.control_sock.recv(4096)
            print(f"[sendToDev] RX: {resp.hex()}")
        except:
            resp = b''

        return resp

    def rx_loop(self):
        """
        계속 1420바이트씩 recvfrom() → processUpdData()로 재조립
        완성된 프레임은 self.frame_queue에 넣기
        """
        while not self.exit_thread:
            try:
                data, addr = self.data_sock.recvfrom(65535)  # UDP 패킷 수신
                total_len = self.processUpdData(data)

                if total_len > 0:  # 완성된 프레임 처리
                    with self.buffer_lock:
                        self.frame_queue.append(self.completedFrame)
                        self.completedFrame = None  # 다음 프레임 준비
            except socket.timeout:
                continue  # 소켓 타임아웃 발생 시 재시도
            except Exception as e:
                print(f"[rx_loop] error: {e}")
                break

    def processUpdData(self, udpPacket: bytes) -> int:
        """
        C++ 'processUpdData()' 유사
        - parse header (number, numPacket, packetNumber, offset, totalSize, payloadSize)
        - copy payload to a reassembly buffer
        - if last packet => return totalSize
          else => return 0
        """

        # 예) 앞 20바이트가 헤더라고 가정 (packetNumber, numPacket 등)
        if len(udpPacket) < 20:
            return 0  # 잘못된 패킷

        number = (udpPacket[0] << 8) | udpPacket[1]
        totalSize = ((udpPacket[2] << 24) | (udpPacket[3] << 16) | (udpPacket[4] << 8) | udpPacket[5])
        payloadSize = ((udpPacket[6] << 8) | udpPacket[7])
        offset = ((udpPacket[8] << 24) | (udpPacket[9] << 16) | (udpPacket[10] << 8) | udpPacket[11])
        numPacket = ((udpPacket[12] << 24) | (udpPacket[13] << 16) | (udpPacket[14] << 8) | udpPacket[15])
        packetNumber = ((udpPacket[16] << 24) | (udpPacket[17] << 16) | (udpPacket[18] << 8) | udpPacket[19])

        # 실제 유효 payload
        # (헤더가 20바이트라고 가정하면, 그 뒤 payloadSize 바이트가 유효)
        if len(udpPacket) < (20 + payloadSize):
            return 0  # 잘못된 패킷
        payload = udpPacket[20: 20 + payloadSize]

        # frameNumber(=number)가 바뀌면, 새 프레임 reassembly 시작
        # 예) self.currentFrameNumber != number -> reset
        if number != self.currentFrameNumber:
            # 새 프레임 시작
            self.currentFrameNumber = number
            self.completedFrame = bytearray(totalSize)
            self.receivedBytes = 0

        # offset 위치에 payload 복사
        self.completedFrame[offset: offset + payloadSize] = payload
        self.receivedBytes += payloadSize

        # 마지막 packet?
        if packetNumber == (numPacket - 1):
            # check totalSize == self.receivedBytes ?
            if self.receivedBytes == totalSize:
                # 완성
                return totalSize
            else:
                # 에러 / mismatch
                # print(f"[processUpdData] Frame incomplete. Expected={totalSize}, Received={self.receivedBytes}")
                self.completedFrame = None  # 불완전 프레임 초기화
                return 0
        else:
            return 0

    # --------------------------------------------------
    # C++의 reqIntegrationTime() 대응
    def reqIntegrationTime(self, int3D=None, hdr1=None, hdr2=None, gray=None):
        """
        C++ NSL3130AA::reqIntegrationTime():
          uint8_t data[] = {0x00, 0x01, ... length=10 ...}
          data[2] = (int3D >> 8) ...
          data[3] = (int3D) ...
          ...
        """
        if int3D is None:  # 지정 안 하면 멤버 변수 값 사용
            int3D = self.integrationTime3D
        if hdr1 is None:
            hdr1 = self.integrationTime3DHdr1
        if hdr2 is None:
            hdr2 = self.integrationTime3DHdr2
        if gray is None:
            gray = self.integrationTimeGrayScale

        data = bytearray(10)  # payload length = 10
        data[0] = 0x00
        data[1] = 0x01  # opcode(0x0001 = SET_INT_TIME 예시)

        # int3D
        data[2] = (int3D >> 8) & 0xFF
        data[3] = int3D & 0xFF

        # hdr1
        data[4] = (hdr1 >> 8) & 0xFF
        data[5] = hdr1 & 0xFF

        # hdr2
        data[6] = (hdr2 >> 8) & 0xFF
        data[7] = hdr2 & 0xFF

        # gray
        data[8] = (gray >> 8) & 0xFF
        data[9] = gray & 0xFF

        self._send_packet(data)

    # --------------------------------------------------
    # C++의 reqHdrMode() 대응
    def reqHdrMode(self, hdr_mode=None):
        """
        uint8_t data[] = {0x00, 0x19, 0x00}; // length=3
        """
        if hdr_mode is None:
            hdr_mode = self.hdr_mode

        data = bytearray(3)
        data[0] = 0x00
        data[1] = 0x19  # opcode(0x0019)
        data[2] = hdr_mode & 0xFF

        self._send_packet(data)

    # --------------------------------------------------
    # C++의 reqMinAmplitude() 대응
    def reqMinAmplitude(self, minAmp=None):
        """
        uint8_t data[] = {0x00, 0x15, 0x00, 0x32}; length=4
        data[2] = hi(minAmp)
        data[3] = lo(minAmp)
        """
        if minAmp is None:
            minAmp = self.minAmplitude

        data = bytearray(4)
        data[0] = 0x00
        data[1] = 0x15  # opcode(0x0015)
        data[2] = (minAmp >> 8) & 0xFF
        data[3] = minAmp & 0xFF

        self._send_packet(data)

    # --------------------------------------------------
    # C++의 reqStopStream() 대응
    def reqStopStream(self):
        """
        uint8_t data[2] = {0x00, 0x06};
        length=2
        """
        data = bytearray(2)
        data[0] = 0x00
        data[1] = 0x06  # opcode(0x0006 = STOP_STREAM)

        self._send_packet(data)

    # --------------------------------------------------
    # C++의 initializeTofcam660() 비슷한 초기화 루틴
    def initializeTofcam660(self):
        """
        예) IntegrationTime / HDR / MinAmplitude / etc. 순서대로 호출
        """
        print("[initializeTofcam660] Start initialization...")
        # 1) reqIntegrationTime(800,100,50,100)
        self.reqIntegrationTime()
        # 2) reqHdrMode(0) (HDR none)
        self.reqHdrMode()
        # 3) reqMinAmplitude(50)
        self.reqMinAmplitude()
        # ... etc ...
        print("[initializeTofcam660] Done.")

    def initialize_device(self):
        """ C++의 initializeTofcam660() 유사 로직 """
        # 아래는 임의 예시. 실제론 initialCode660[]을 sendToDev()로 전송
        # integrationTime, ROI, Overflow, etc...
        print("initialize_device() - sending initial commands...")
        # 예시
        self.reqIntegrationTime()
        self.req_min_amplitude(self.minAmplitude)
        # HDR, Filter 등도 여기서 설정
        # ...

    def req_min_amplitude(self, min_amp):
        """C++: reqMinAmplitude()"""
        data = bytearray([0x00, 0x15])
        data.append((min_amp >> 8) & 0xFF)
        data.append(min_amp & 0xFF)
        self._send_packet(bytes(data))
        print(f"req_min_amplitude: {min_amp}")


    def reqStreamingFrame(self):
        """
        C++: reqStreamingFrame()
        예) 3바이트 payload [0x00, 0x02, 0x01]
        opcode=0x0002, 3rd byte=1 => streaming measurement
        """
        data = bytearray(3)
        data[0] = 0x00
        data[1] = 0x02  # opcode=0x0002
        data[2] = 0x01  # 1=streaming, etc.

        self._send_packet(data)

    def reqOverflow(self, adcOverflow: bool, saturation: bool):
        """
        C++: reqOverflow()
         4바이트 payload [0x00,0x0A, X, Y]
         X=1 or 0, Y=1 or 0
        """
        data = bytearray(4)
        data[0] = 0x00
        data[1] = 0x0A  # opcode=0x000A
        data[2] = 1 if adcOverflow else 0
        data[3] = 1 if saturation else 0

    # --------------------------------------------------
    # C++의 startCaptureCommand() 비슷한 함수
    def startCaptureCommand(self):
        """
        C++: startCaptureCommand() 안에서
             reqIntegrationTime(), reqMinAmplitude(), reqFilterParameter(), etc.
             마지막으로 reqStreamingFrame()까지 호출해 '스트리밍 시작' 명령.
        """
        print("[startCaptureCommand] Setting integration times, etc...")
        # (1) IntegrationTime
        self.reqIntegrationTime()
        # (2) HDR Mode
        self.reqHdrMode()
        # (3) MinAmplitude
        self.reqMinAmplitude()
        # (4) Overflow disable
        self.reqOverflow(False, False)
        # ... 필요하다면 Filter, ROI, Compensation 등등 ...
        # (마지막) 스트리밍 프레임 명령
        self.reqStreamingFrame()
        print("[startCaptureCommand] done. (Streaming requested)")

    def capture(self, timeout=3.0):
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.buffer_lock:
                if self.frame_queue:
                    raw_data = self.frame_queue.pop(0)  # 재조립된 한 프레임
                    img_frame = self.parse_frame(raw_data)
                    return img_frame
        return None  # 타임아웃

    def parse_frame(self, raw_data: bytes) -> ImageFrame:
        """
        C++의 getDistanceAmplitude(), getGrayscaled() 등에 해당.
        여기서는 raw_data를 직접 파싱해야 함.
        예) distance+amplitude: 4 bytes/pixel
        """
        # 예시: (320*240*4) + 헤더 정도라 가정
        # 실제로는 C++의 header (RxHeader), offset, etc.를 해석해야 함
        if len(raw_data) < (320 * 240 * 4):
            # 잘못된 패킷(멀티 패킷 조립 필요) -> 여기선 생략
            pass

        # 간단히 numpy 변환 (예시)
        # raw_data[헤더:]부터 320*240*4 짜리 픽셀 데이터
        # dist: 2바이트, amp: 2바이트 -> pixelDistance, pixelAmplitude
        frame_cv2 = ImageFrame()

        # distance / amplitude 행렬
        distance_map = np.zeros((240, 320), dtype=np.uint16)
        amplitude_map = np.zeros((240, 320), dtype=np.uint16)

        offset = 0  # 예시
        for y in range(240):
            for x in range(320):
                # dist(2 byte) + amp(2 byte)
                d0 = raw_data[offset]
                d1 = raw_data[offset + 1]
                dist_val = (d1 << 8) | d0

                a0 = raw_data[offset + 2]
                a1 = raw_data[offset + 3]
                amp_val = (a1 << 8) | a0

                distance_map[y, x] = dist_val
                amplitude_map[y, x] = amp_val

                offset += 4

        # 컬러맵으로 만들기 (예시, OpenCV applyColorMap)
        # uint16 -> float -> normalize -> uint8
        dist_vis = self.apply_distance_colormap_lut(distance_map, self.colorLUT, max_dist=self.maxDist)
        amp_vis = self.apply_amplitude_colormap(amplitude_map)

        frame_cv2.distMat = dist_vis
        frame_cv2.frameMat = amp_vis
        frame_cv2.isRotate = (self.rotate_90 == 1)

        return frame_cv2
        # """
        # 벡터화를 통해 distance_map 및 amplitude_map을 효율적으로 처리
        # """
        # num_pixels = 320 * 240  # 예상되는 픽셀 수
        # total_bytes = num_pixels * 4
        #
        # # 데이터 길이 확인
        # if len(raw_data) < total_bytes:
        #     raise ValueError(f"Incomplete frame data. Received: {len(raw_data)}, Expected: {total_bytes}")
        #
        # # Numpy 배열로 변환
        # raw_array = np.frombuffer(raw_data[:total_bytes], dtype=np.uint8)  # 길이 초과 방지
        #
        # # 데이터 벡터화 변환
        # distance_map = raw_array[0::4] + (raw_array[1::4] << 8)
        # amplitude_map = raw_array[2::4] + (raw_array[3::4] << 8)
        #
        # # 2D 배열로 변환
        # distance_map = distance_map.reshape((240, 320))
        # amplitude_map = amplitude_map.reshape((240, 320))
        #
        # frame_cv2 = ImageFrame()
        # frame_cv2.distMat = self.apply_distance_colormap(distance_map)
        # frame_cv2.frameMat = self.apply_amplitude_colormap(amplitude_map)
        # return frame_cv2

    def apply_distance_colormap(self, dist_map: np.ndarray) -> np.ndarray:
        """C++: setDistanceColor() 대체 예시"""
        # dist_map(0~maxDistance) -> 0~255 로 normalize
        dist_norm = np.clip(dist_map, 0, self.maxDist).astype(np.float32) / self.maxDist
        dist_norm = (dist_norm * 255).astype(np.uint8)

        dist_vis = cv2.applyColorMap(dist_norm, self.color_map)
        return dist_vis

    def apply_amplitude_colormap(self, amp_map: np.ndarray) -> np.ndarray:
        """
            Converts amplitude data to a grayscale image (0-255).
            Handles unexpected amplitude ranges by dynamically adjusting the max value.
            """
        # 동적 범위 계산
        max_amp = np.max(amp_map)  # 데이터의 실제 최대값 사용
        min_amp = np.min(amp_map)  # 데이터의 실제 최소값 사용

        if max_amp == min_amp:  # 데이터가 단일 값일 경우 방어 처리
            max_amp += 1
        # 정규화 (0~255)
        amp_normalized = ((amp_map - min_amp) / (max_amp - min_amp) * 255.0).astype(np.uint8)
        # Grayscale -> BGR 변환
        amp_bgr = cv2.cvtColor(amp_normalized, cv2.COLOR_GRAY2BGR)
        return amp_bgr

    def read(self):
        """
        Provide a (success, frame) interface like cv2.VideoCapture.read().
        frame will be a numpy array (H, W, 3) suitable for YOLO.
        """
        framestruct = self.capture(timeout=2.0)  # or however long
        if framestruct is None:
            return False, None

        # framestruct is an ImageFrame object
        #   framestruct.frameMat => amplitude color image
        #   framestruct.distMat  => distance color image

        # YOLO는 일반 RGB(BGR) 3채널 영상을 기대하므로
        # 일단 amplitude color를 쓰겠다고 가정:
        # out_frame = framestruct.distMat  # shape (H,W,3), dtype=uint8
        out_frame = framestruct.frameMat

        # OR if you want distance or combine them, etc.
        # out_frame = framestruct.distMat

        # return in (bool, np.ndarray) style
        return True, out_frame

    def apply_distance_colormap_lut(self, distance_map: np.ndarray, color_lut, max_dist=30000):
        """
        distance_map: 2D array of distance in mm
        color_lut: list of (B,G,R) up to max_dist steps
        max_dist: LUT 범위
        return: colored image (H,W,3)
        """
        H, W = distance_map.shape
        out_img = np.zeros((H, W, 3), dtype=np.uint8)

        for y in range(H):
            for x in range(W):
                dist_val = distance_map[y, x]
                if dist_val < 0:
                    dist_val = 0
                elif dist_val >= max_dist:  # '=' 포함
                    dist_val = max_dist - 1  # 인덱스는 (0 ~ max_dist-1)

                # index = dist_val (0~max_dist)
                (b,g,r) = color_lut[dist_val]
                out_img[y,x] = (b,g,r)
        return out_img
        # """
        # 벡터화된 LUT 컬러맵 적용
        # """
        # clipped_dist = np.clip(distance_map, 0, max_dist - 1)  # Ensure values are within LUT range
        # lut_indices = clipped_dist.astype(np.int32)
        # colored_image = np.array(color_lut, dtype=np.uint8)[lut_indices]
        #
        # return colored_image

    def interpolate_py(self, x, x0, y0, x1, y1):
        """
        C++의 interpolate()와 동일한 선형보간.
        x: 현재 값
        (x0, y0) ~ (x1, y1) 구간을 보간
        """
        if x1 == x0:
            return y0
        return (x - x0) * (y1 - y0) / (x1 - x0) + y0

    def create_color_map_pixel(self, num_steps, idx):
        """
        C++ NSL3130AA::createColorMapPixel()와 유사한 파이썬 함수
        idx: 0 ~ num_steps-1
        return (red, green, blue): 0~255 범위의 정수 튜플
        """
        k = 1.0
        BIT0 = -0.125 * k - 0.25
        BIT1 = BIT0 + 0.25 * k
        BIT2 = BIT1 + 0.25 * k
        BIT3 = BIT2 + 0.25 * k

        G0 = BIT1
        G1 = G0 + 0.25 * k
        G2 = G1 + 0.25 * k
        G3 = G2 + 0.25 * k + 0.125

        R0 = BIT2
        R1 = R0 + 0.25 * k
        R2 = R1 + 0.25 * k
        R3 = R2 + 0.25 * k + 0.25

        i = float(idx) / float(num_steps) - 0.25 * k

        # red
        if R0 <= i < R1:
            red = int(self.interpolate_py(i, R0, 0, R1, 255))
        elif R1 <= i < R2:
            red = 255
        elif R2 <= i < R3:
            red = int(self.interpolate_py(i, R2, 255, R3, 0))
        else:
            red = 0

        # green
        if G0 <= i < G1:
            green = int(self.interpolate_py(i, G0, 0, G1, 255))
        elif G1 <= i < G2:
            green = 255
        elif G2 <= i < G3:
            green = int(self.interpolate_py(i, G2, 255, G3, 0))
        else:
            green = 0

        # blue
        if BIT0 <= i < BIT1:
            blue = int(self.interpolate_py(i, BIT0, 0, BIT1, 255))
        elif BIT1 <= i < BIT2:
            blue = 255
        elif BIT2 <= i < BIT3:
            blue = int(self.interpolate_py(i, BIT2, 255, BIT3, 0))
        else:
            blue = 0

        return (red, green, blue)

    def build_color_lut(self, num_steps=30000):
        """
        예: 30000단계 컬러맵 (거리 최대 30000mm 가정)
        return: list/array of (B, G, R) for each step
        """
        color_lut = []
        for idx in range(num_steps):
            (r, g, b) = self.create_color_map_pixel(num_steps, idx)
            color_lut.append((b, g, r))  # OpenCV는 BGR 순
        return color_lut

if __name__ == "__main__":
    # 1) 객체 생성
    lidar = NSL3130AA(ipaddr="192.168.241.254", port=50660)
    # 2) TCP 연결
    lidar.connect()
    # 3) 초기화 시퀀스
    lidar.startCaptureCommand()

    try:
        while True:
            frame = lidar.capture(timeout=2.0)
            if frame is not None:
                # frame.frameMat, frame.distMat → cv2.imshow
                cv2.imshow("Amplitude(or Gray)", frame.frameMat)
                cv2.imshow("Distance", frame.distMat)

                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    break
            else:
                print("no frame (timeout)")

    except KeyboardInterrupt:
        pass
    finally:
        if lidar:
            lidar.reqStopStream()
            lidar.close()
        cv2.destroyAllWindows()
        lidar.reqStopStream()