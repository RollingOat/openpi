import depthai as dai
import cv2

def createPipeline(pipeline, rgb_resolution=(640, 480), frame_rate=20):
    camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    output = camRgb.requestOutput(rgb_resolution, dai.ImgFrame.Type.NV12 ,dai.ImgResizeMode.CROP, frame_rate).createOutputQueue()
    return pipeline, output

def start_multiple_oak_cameras():
    """Starts multiple OAK camera captures."""
    deviceInfos = dai.Device.getAllAvailableDevices()
    print("=== Found devices: ", deviceInfos)
    queues = []
    pipelines = []

    for deviceInfo in deviceInfos:
        pipeline = dai.Pipeline()
        device = pipeline.getDefaultDevice()
        
        print("===Connected to ", deviceInfo.getDeviceId())
        
        mxId = device.getDeviceId()
        cameras = device.getConnectedCameras()
        usbSpeed = device.getUsbSpeed()
        eepromData = device.readCalibration2().getEepromData()
        print("   >>> Device ID:", mxId)
        print("   >>> Num of cameras:", len(cameras))
        if eepromData.boardName != "":
            print("   >>> Board name:", eepromData.boardName)
        if eepromData.productName != "":
            print("   >>> Product name:", eepromData.productName)
        
        pipeline, output = createPipeline(pipeline)
        pipeline.start()
        pipelines.append(pipeline)

        queues.append(output)
    return queues, pipelines, deviceInfos

def get_images_from_multiple_oak_cameras(queues, deviceInfos):
    images = {}
    for i, q in enumerate(queues):
        videoIn = q.get()
        images[deviceInfos[i].getDeviceId()] = videoIn.getCvFrame()
    return images


def main():
    queues, pipelines, deviceInfos = start_multiple_oak_cameras()
    try:
        while True:
            images = get_images_from_multiple_oak_cameras(queues, deviceInfos)
            for i, (mxId, frame) in enumerate(images.items()):
                cv2.imshow(f"OAK Camera {mxId}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        for pipeline in pipelines:
            pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()