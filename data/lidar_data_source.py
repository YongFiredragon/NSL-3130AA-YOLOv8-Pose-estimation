import socket
import struct
import threading
import time
import numpy as np
import cv2

# 필요하다면: from pyserial import serial  (시리얼 통신이 필요하면)

# 장치 초기 세팅에 필요한 상수들
START_MARKER = b'\xff\xff\xaa\x55'
END_MARKER   = b'\xff\xff\x55\xaa'

# 해상도 / 버퍼 관련
LIDAR_IMAGE_WIDTH = 320
LIDAR_IMAGE_HEIGHT = 240

NSL3130_BUFF_SIZE = 308000  # C++ 코드에 정의되어 있던 최대 패킷 버퍼
PACKET_INFO_SIZE  = 8       # start_marker(4) + payload_length(4)
TAIL_PREFIX_SIZE  = 4       # end_marker(4)

# LiDAR 모드 상수 (C++ enum TOFCAM660_DATATYPE / tofcamModeType 등)
AMPLITEDE_DISTANCE_MODE     = 0
AMPLITEDE_DISTANCE_EX_MODE  = 1
DISTANCE_GRAYSCALE_MODE     = 2
GRAYSCALE_MODE              = 3
DISTANCE_MODE               = 4  # 예시, C++에선 COMMAND_GET_DISTANCE=3

# HDR 모드 (예: HDR_NONE_MODE=0, HDR_SPATIAL_MODE=1, HDR_TEMPORAL_MODE=2)
HDR_NONE_MODE = 0
HDR_SPATIAL_MODE = 1
HDR_TEMPORAL_MODE = 2

COMMANDS = {
    # 예) SET_INT_TIME: opcode=0x0001, 필드 4개(각 2바이트)
    "SET_INT_TIME": {
        "opcode": 0x0001,
        "fields": [
            {"name": "intTime3D",       "size": 2},  # 2 bytes
            {"name": "intTime3DHdr1",   "size": 2},
            {"name": "intTime3DHdr2",   "size": 2},
            {"name": "intTimeGray",     "size": 2},
        ]
    },
    # 예) SET_HDR: opcode=0x0019, 필드 1개(1바이트)
    "SET_HDR": {
        "opcode": 0x0019,
        "fields": [
            {"name": "hdrMode", "size": 1},
        ]
    },
    # 예) SET_MIN_AMPLITUDE: opcode=0x0015, 필드 1개(2바이트)
    "SET_MIN_AMPLITUDE": {
        "opcode": 0x0015,
        "fields": [
            {"name": "minAmp", "size": 2},
        ]
    },
    # 예) SET_ADC_OVERFLOW: opcode=0x000A, 필드 2바이트
    #    (adcOverflowEnable(1bit?), saturationEnable(1bit?) 등 구체 구조는 프로젝트에 맞게)
    "SET_ADC_OVERFLOW": {
        "opcode": 0x000A,
        "fields": [
            {"name": "adcOverFlow", "size": 1},
            {"name": "saturation",  "size": 1},
        ]
    },
    # 그 외 SET_COMPENSATION, SET_ROI, SET_DUALBEAM, etc.
    # 필요 시 전부 등록
    # ...
    # 마지막으로 GET_DIST_AMPLITUDE(스트리밍 요청) 같은 명령도 추가
    "GET_DIST_AMPLITUDE": {
        "opcode": 0x0002,
        "fields": [
            {"name": "streamMode", "size": 1},  # 예) 1=streaming
        ]
    }
}

VALUE_STREAMING_MEASUREMENT = 1

class CaptureOptions:
    """ C++의 CaptureOptions 구조체를 Python으로 대략 포팅 """
    def __init__(self, ipaddr="192.168.241.254"):
        self.ipaddr = ipaddr
        self.lidarType             = 1
        self.captureType           = AMPLITEDE_DISTANCE_EX_MODE
        self.integrationTime       = 800
        self.grayIntegrationTime   = 100
        self.maxDistance           = 12500
        self.minAmplitude          = 50
        self.detectDistance        = 0
        self.edgeThresHold         = 0
        self.medianFilterSize      = 0
        self.medianFilterEnable    = 0
        self.averageFilterEnable   = 0
        self.temporalFilterFactorActual = 0
        self.temporalFilterThreshold = 0
        self.interferenceUseLashValueEnable = 0
        self.interferenceLimit     = 0
        self.dualbeamState         = 0
        self.hdr_mode              = HDR_NONE_MODE
        self.deeplearning          = 0
        self.modelType             = 0
        # 기타 필요한 필드들...

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
    C++의 NSL3130AA 클래스(소켓 기반) 포팅 예시.
    - 실제 동작을 위해서는 sendToDev(), recvFromTcp(), processUpdData() 등 세부 구현 필요.
    """

    def __init__(self, ipaddr="192.168.241.254"):
        self.ipaddr         = ipaddr
        self.control_sock   = None  # TCP
        self.data_sock      = None  # UDP
        self.exit_thread    = False
        self.thread         = None

        # 버퍼 (C++의 ring-buffer 구조를 간단화)
        self.buffer_lock = threading.Lock()
        self.frame_queue = []

        # tofcamInfo 구조체 대체
        # (C++에선 TOFCAM660_INFO, TOFCAM660_DATABUF, etc.)
        self.tofcamModeType = AMPLITEDE_DISTANCE_EX_MODE
        self.rotate_90      = 0
        self.imageWidth     = 320
        self.imageHeight    = 240

        # 기타 설정들
        self.integrationTime3D        = 800
        self.integrationTimeGrayScale = 100
        self.hdr_mode                 = HDR_NONE_MODE
        self.minAmplitude             = 50

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
        """ 스트리밍 종료, 스레드 중지, 소켓 종료 """
        self.exit_thread = True
        # C++: reqStopStream(control_sock)
        self.req_stop_stream()
        if self.thread and self.thread.is_alive():
            self.thread.join()

        if self.control_sock:
            self.control_sock.close()
            self.control_sock = None
        if self.data_sock:
            self.data_sock.close()
            self.data_sock = None

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

    def rx_loop(self):
        """UDP 수신 루프 (C++의 rxTofcam660에서 rxSocket() 부분)"""
        while not self.exit_thread:
            try:
                data, addr = self.data_sock.recvfrom(1500)
                print(f"[rx_loop] received {len(data)} bytes from {addr}")
                # 여기서 data를 frame_queue에 append하는지, 파싱이 필요한지...
            except socket.timeout:
                print("[rx_loop] timeout")
                continue
            except Exception as e:
                print(f"[rx_loop] error: {e}")
                break

    def capture(self, timeout=3.0):
        """
        C++: Capture(void** image, int timeout)
        - ring-buffer에 새 프레임이 쌓일 때까지 대기
        - 도달하면 파싱 + Mat 변환
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.buffer_lock:
                if self.frame_queue:
                    raw_data = self.frame_queue.pop(0)
                    # parse + convert
                    img_frame = self.parse_frame(raw_data)
                    return img_frame
            time.sleep(0.001)
        return None  # 타임아웃

    def parse_frame(self, raw_data: bytes) -> ImageFrame:
        """
        C++의 getDistanceAmplitude(), getGrayscaled() 등에 해당.
        여기서는 raw_data를 직접 파싱해야 함.
        예) distance+amplitude: 4 bytes/pixel
        """
        # 예시: (320*240*4) + 헤더 정도라 가정
        # 실제로는 C++의 header (RxHeader), offset, etc.를 해석해야 함
        if len(raw_data) < (320*240*4):
            # 잘못된 패킷(멀티 패킷 조립 필요) -> 여기선 생략
            pass

        # 간단히 numpy 변환 (예시)
        # raw_data[헤더:]부터 320*240*4 짜리 픽셀 데이터
        # dist: 2바이트, amp: 2바이트 -> pixelDistance, pixelAmplitude
        frame = ImageFrame()

        # distance / amplitude 행렬
        distance_map  = np.zeros((240,320), dtype=np.uint16)
        amplitude_map = np.zeros((240,320), dtype=np.uint16)

        offset = 0  # 예시
        for y in range(240):
            for x in range(320):
                # dist(2 byte) + amp(2 byte)
                d0 = raw_data[offset]
                d1 = raw_data[offset+1]
                dist_val = (d1 << 8) | d0

                a0 = raw_data[offset+2]
                a1 = raw_data[offset+3]
                amp_val = (a1 << 8) | a0

                distance_map[y,x]  = dist_val
                amplitude_map[y,x] = amp_val

                offset += 4

        # 컬러맵으로 만들기 (예시, OpenCV applyColorMap)
        # uint16 -> float -> normalize -> uint8
        dist_vis = self.apply_distance_colormap(distance_map)
        amp_vis  = self.apply_amplitude_colormap(amplitude_map)

        frame.distMat  = dist_vis
        frame.frameMat = amp_vis
        frame.isRotate = (self.rotate_90 == 1)

        return frame

    def apply_distance_colormap(self, dist_map: np.ndarray) -> np.ndarray:
        """C++: setDistanceColor() 대체 예시"""
        # dist_map(0~maxDistance) -> 0~255 로 normalize
        max_dist = 12500  # 임의
        dist_norm = np.clip(dist_map, 0, max_dist).astype(np.float32) / max_dist
        dist_norm = (dist_norm * 255).astype(np.uint8)

        dist_vis = cv2.applyColorMap(dist_norm, self.color_map)
        return dist_vis

    def apply_amplitude_colormap(self, amp_map: np.ndarray) -> np.ndarray:
        """C++: setAmplitudeColor() 대체 예시"""
        max_amp = 3000  # 임의
        amp_norm = np.clip(amp_map, 0, max_amp).astype(np.float32) / max_amp
        amp_norm = (amp_norm * 255).astype(np.uint8)

        amp_vis = cv2.applyColorMap(amp_norm, self.color_map)
        return amp_vis

    # 아래부턴 C++의 reqXXX() 함수 예시

    def sendToDev(self, sock: socket.socket, data: bytes) -> bytes:
        """
        C++ 'sendToDev(...)'에 대응
        - startMarker(4B)
        - length(4B, Big Endian) = len(data)
        - payload(data)
        - endMarker(4B)
        """

        length = len(data)
        length_bytes = struct.pack('>I', length)  # Big Endian 4바이트

        pkt = START_MARKER + length_bytes + data + END_MARKER
        sock.sendall(pkt)

        print(f"[send_packet] TX= {pkt.hex().upper()}")

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


    def req_stop_stream(self):
        """C++: reqStopStream()"""
        payload = b'\x00\x06'  # COMMAND_STOP_STREAM (예시)
        self.sendToDev(payload)
        print("reqStopStream sent")

    def reqIntegrationTime(sock, integrationTime3D, integrationTime3DHdr1, integrationTime3DHdr2,
                           integrationTimeGrayScale):
        """
        C++ 'reqIntegrationTime()'과 구조/배열 하드코딩을 비슷하게 맞춘 예시
        """
        data = bytearray(10)  # length=10, opcode=0x00,0x01
        data[0] = 0x00
        data[1] = 0x01

        # integrationTime3D
        data[2] = (integrationTime3D >> 8) & 0xFF
        data[3] = (integrationTime3D) & 0xFF

        # integrationTime3DHdr1
        data[4] = (integrationTime3DHdr1 >> 8) & 0xFF
        data[5] = (integrationTime3DHdr1) & 0xFF

        # integrationTime3DHdr2
        data[6] = (integrationTime3DHdr2 >> 8) & 0xFF
        data[7] = (integrationTime3DHdr2) & 0xFF

        # integrationTimeGrayScale
        data[8] = (integrationTimeGrayScale >> 8) & 0xFF
        data[9] = (integrationTimeGrayScale) & 0xFF

        sendToDev(sock, data)  # 전송함수


    def req_min_amplitude(self, min_amp):
        """C++: reqMinAmplitude()"""
        data = bytearray([0x00, 0x15])
        data.append((min_amp >> 8) & 0xFF)
        data.append(min_amp & 0xFF)
        self.sendToDev(bytes(data))
        print(f"req_min_amplitude: {min_amp}")

    def start_capture_command(self, camOpt: CaptureOptions):
        """
        C++: startCaptureCommand(...)
        명령들 보내고, 모드 설정, 스트리밍 시작 등
        """
        self.tofcamModeType = camOpt.captureType
        self.integrationTime3D = camOpt.integrationTime
        self.integrationTimeGrayScale = camOpt.grayIntegrationTime
        self.hdr_mode = camOpt.hdr_mode
        self.minAmplitude = camOpt.minAmplitude
        # etc...

        # 실제로는 reqIntegrationTime(), reqHdrMode(), reqFilterParameter(), etc.
        self.reqIntegrationTime()
        self.req_min_amplitude(self.minAmplitude)
        self.reqHdrMode()
        self.reqLedControl()
        self.reqStreamingFrame()
        # 마지막에 reqStreamingFrame() 같은 거 호출
        # ...

    def reqStreamingFrame(self):
        """
        C++: reqStreamingFrame()에 해당하는 로직을 Python으로 옮긴 예.
        """
        # 예: distance+amplitude = 0x02
        # tofcamModeType에 따라 cmdType 결정
        cmdType = self.getCommandByType()

        # data = [0x00, 0x02, VALUE_STREAMING_MEASUREMENT] 이런 식으로 구성
        # 단, 실제론 C++처럼 바이트 배열을 만들어야 하므로 bytearray로 작성
        payload = bytearray([0x00, 0x02, VALUE_STREAMING_MEASUREMENT])  # 0x03이 VALUE_STREAMING_MEASUREMENT라고 가정
        payload[1] = cmdType   # captureType에 따라 opcode를 덮어씀

        # 전송 (sendToDev와 유사한 함수 사용)
        resp = self.sendToDev(payload)
        print(f"reqStreamingFrame: resp len = {len(resp)}")

    def getCommandByType(modeType: int) -> int:
        """
        C++의 NSL3130AA::getCommandByType() 대응 파이썬 버전.
        modeType(캡처 모드)에 따라 서로 다른 opcode(명령 코드)를 반환한다.
        """
        if modeType in (AMPLITEDE_DISTANCE_MODE, AMPLITEDE_DISTANCE_EX_MODE):
            return 0x02  # distance+amplitude
        elif modeType == DISTANCE_MODE:
            return 0x03
        elif modeType == DISTANCE_GRAYSCALE_MODE:
            return 0x08  # distance+grayscale
        elif modeType == GRAYSCALE_MODE:
            return 0x05  # pure grayscale
        else:
            return 0x00  # default/fallback

    def reqHdrMode(self):
        """
        C++: reqHdrMode()
        예: payload = [0x00, 0x19, hdr_mode], 총 3바이트
        """
        data_len = 3
        data = bytearray(data_len)

        data[0] = 0x00
        data[1] = 0x19
        data[2] = self.hdr_mode  # 0=HDR_NONE, 1=HDR_SPATIAL, 2=HDR_TEMPORAL

        resp = self.sendToDev(data)
        print(f"[reqHdrMode] hdr_mode={self.hdr_mode}, len(resp)={len(resp)}")

    def reqLedControl(self):
        """
        C++: reqGrayscaleLedControl(SOCKET control_sock, int ledOnOff)
        예: payload = [0x00, 0x27, led_control]
        """
        data_len = 3
        data = bytearray(data_len)
        data[0] = 0x00
        data[1] = 0x27
        data[2] = self.led_control  # 1=on, 0=off

        resp = self.sendToDev(data)
        print(f"[reqLedControl] led_control={self.led_control}, len(resp)={len(resp)}")

    def build_payload(cmd_name: str, params: dict) -> bytes:
        """
        1) COMMANDS 사전에서 cmd_name에 해당하는 opcode와 fields를 찾는다.
        2) opcode(2바이트) + 각 field를 순서대로 붙여서 payload를 만든다.
        3) 반환: payload(바이트열)
        """
        import struct

        cmd_info = COMMANDS[cmd_name]
        opcode = cmd_info["opcode"]  # 예: 0x0001
        fields = cmd_info["fields"]

        payload = bytearray()

        # --- 2바이트 opcode ---
        payload.append((opcode >> 8) & 0xFF)  # high byte
        payload.append(opcode & 0xFF)  # low byte

        # --- fields 순회하며 bytearray 확장 ---
        for f in fields:
            field_name = f["name"]
            field_size = f["size"]
            value = params[field_name]  # 유저가 넣어준 파라미터 값

            if field_size == 1:
                # 1바이트
                payload.append(value & 0xFF)
            elif field_size == 2:
                # 2바이트 (High Byte -> Low Byte)
                payload.append((value >> 8) & 0xFF)
                payload.append(value & 0xFF)
            else:
                raise NotImplementedError(f"field_size={field_size} not handled")

        return bytes(payload)

# 만약 메인 스크립트처럼 쓰고 싶다면:
if __name__ == "__main__":
    camOpt = CaptureOptions()
    camOpt.captureType = AMPLITEDE_DISTANCE_EX_MODE
    # camOpt.integrationTime = 1000
    # ...

    nslCam = NSL3130AA("192.168.241.254")
    nslCam.connect()
    nslCam.start_capture_command(camOpt)

    try:
        while True:
            frame = nslCam.capture(timeout=2.0)
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
        nslCam.close()
        cv2.destroyAllWindows()