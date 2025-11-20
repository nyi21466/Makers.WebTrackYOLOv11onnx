import "./assets/App.css";
import cv from "@techstark/opencv-js";
import { useEffect, useRef, useState, useCallback } from "react";
import { model_loader } from "./utils/model_loader";
import { inference_pipeline } from "./utils/inference_pipeline";
import { draw_bounding_boxes } from "./utils/draw_bounding_boxes";
import { BYTETracker } from "./utils/tracker";
import classes from "./utils/yolo_classes.json";
import Marquee from "react-fast-marquee";

// TODO: add set class.json

// set Components
function SettingsPanel({
  backendSelectorRef,
  modelSelectorRef,
  cameraSelectorRef,
  imgszTypeSelectorRef,
  cameras,
  customModels,
  onModelChange,
  activeFeature,
  updateImgszTypeLock,
}) {
  return (
    <center>
    <div
      style={{color: "black", fontWeight: "bold", justifyContent: "center", border: "2px solid #231e09ff"}}
      id="setting-container"
      className="container text-lg flex flex-col md:flex-row md:justify-evenly gap-2 md:gap-6"
    >
      <div id="selector-container">
        <label>Backend:</label>
        <select
          name="device-selector"
          ref={backendSelectorRef}
          onChange={onModelChange}
          disabled={activeFeature !== null}
          className="ml-2"
        >
          <option value="wasm">WASM(CPU)</option>
          <option value="webgpu">WebGPU</option>
        </select>
      </div>
      <div id="selector-container">
        <label>Model:</label>
        <select
          name="model-selector"
          ref={modelSelectorRef}
          // onChange={onModelChange}
          onChange={(e) => {
            onModelChange(e);
            // call lock immediately with the new value
            updateImgszTypeLock?.(e.target.value);
          }}
          disabled={activeFeature !== null}
          className="ml-2"
        >
          <option value="yolo11n">YOLO11n-10.1M</option>
          <option value="yolo11s">YOLO11s-36.1M</option>
          <option value="yolov8n">YOLOv8n-12.2M</option>
          <option value="yolov8s">YOLOv8s-42.7M</option>
          {/* <option value="your-custom-model">Your Custom Model</option> */}
          {customModels.map((model, index) => (
            <option key={index} value={model.url}>
              {model.name}
            </option>
          ))}
        </select>
      </div>
      <div id="selector-container">
        <label>Camera:</label>
        <select
          ref={cameraSelectorRef}
          disabled={activeFeature !== null}
          className="ml-2"
        >
          {cameras.map((camera, index) => (
            <option key={index} value={camera.deviceId}>
              {camera.label || `Camera ${index + 1}`}
            </option>
          ))}
        </select>
      </div>
      <div id="selector-container">
        <label>Imgsz_type:</label>
        <select
          disabled={activeFeature !== null}
          ref={imgszTypeSelectorRef}
          className="ml-2"
        >
          <option value="dynamic">Dynamic</option>
          <option value="zeroPad">Zero Pad</option>
        </select>
      </div>
    </div>
    </center>
  );
}

// Display Components
function ImageDisplay({
  cameraRef,
  imgRef,
  overlayRef,
  videoRef,
  imgSrc,
  onCameraLoad,
  onImageLoad,
  onVideoLoad,
  activeFeature,
}) {
  return (
    <center>
    <div 
    style={{justifyContent: "center", border: "2px solid #231e09ff"}}
    className="container shadow-lg relative min-h-[320px] flex justify-center items-center">
      <video
        ref={videoRef}
        onLoadedMetadata={onVideoLoad}
        hidden={activeFeature !== "video"}
        autoPlay
        loop
        className="block md:max-w-[720px] max-h-[640px] rounded-lg"
      />
      <video
        className="block md:max-w-[720px] max-h-[640px] rounded-lg mx-auto"
        ref={cameraRef}
        onLoadedMetadata={onCameraLoad}
        hidden={activeFeature !== "camera"}
        autoPlay
      />
      <img
        id="img"
        ref={imgRef}
        src={imgSrc}
        onLoad={onImageLoad}
        hidden={activeFeature !== "image"}
        className="block md:max-w-[720px] max-h-[640px] rounded-lg"
      />
      <canvas
        ref={overlayRef}
        hidden={activeFeature === null}
        className="absolute"
      />
    </div>
    </center>
  );
}

// button Components
function ControlButtons({
  imgSrc,
  fileVideoRef,
  fileImageRef,
  handle_OpenVideo,
  handle_CloseVideo,
  handle_OpenImage,
  handle_ToggleCamera,
  handle_AddModel,
  activeFeature,
}) {
  return (
    <center>
    <div id="btn-container" 
    style={{fontSize: "13px", justifyContent: "center", border: "2px solid #231e09ff"}}
    className="container flex justify-around gap-x-4">
      <input
        type="file"
        accept="video/mp4"
        hidden
        ref={fileVideoRef}
        onChange={(e) => {
          if (e.target.files[0]) {
            handle_OpenVideo(e.target.files[0]);
            e.target.value = null;
          }
        }}
      />

      <button
        className="btn"
        onClick={() => {
          if (activeFeature === "video") {
            // Close video
            handle_CloseVideo();
          } else {
            // Open video file dialog
            fileVideoRef.current.click();
          }
        }}
        disabled={activeFeature !== null && activeFeature !== "video"}
      >
        {activeFeature === "video" ? "Close Video" : "Open Video"}
      </button>

      <input
        type="file"
        accept="image/*"
        hidden
        ref={fileImageRef}
        onChange={(e) => {
          if (e.target.files[0]) {
            const file = e.target.files[0];
            const imgUrl = URL.createObjectURL(file);
            handle_OpenImage(imgUrl);
            e.target.value = null;
          }
        }}
      />

      <button
        className="btn"
        onClick={() =>
          imgSrc ? handle_OpenImage() : fileImageRef.current.click()
        }
        disabled={activeFeature !== null && activeFeature !== "image"}
      >
        {activeFeature === "image" ? "Close Image" : "Open Image"}
      </button>

      <button
        className="btn"
        onClick={handle_ToggleCamera}
        disabled={activeFeature !== null && activeFeature !== "camera"}
      >
        {activeFeature === "camera" ? "Close Camera" : "Open Camera"}
      </button>

      <label className="btn">
        <input type="file" accept=".onnx" onChange={handle_AddModel} hidden />
        <span>Add model</span>
      </label>
    </div>
    </center>
  );
}

// model status Components
function ModelStatus({ warnUpTime, inferenceTime, statusMsg, statusColor }) {
  return (
    <div id="model-status-container" className="text-xl md:text-2xl px-2">
      <div
        id="inferenct-time-container"
        className="flex flex-col md:flex-row md:justify-evenly text-lg md:text-xl my-3 md:my-3"
      >
        <p className="mb-2 md:mb-0 text-blue-400">
          Warm up time: <span className="text-lime-500">{warnUpTime}ms</span>
        </p>
        <p className="text-blue-400">
          Inference time:{" "}
          <span className="text-lime-500">{inferenceTime}ms</span>
        </p>
      </div>
      <p
        className={statusColor !== "green" ? "animate-text-loading" : ""}
        style={{ color: statusColor }}
      >
        {statusMsg}
      </p>
    </div>
  );
}

function ResultsTable({ details }) {
  if (details.length === 0) {
    return (
      <details className="text-gray-600 group px-2">
        <summary className="my-2 hover:text-gray-600 cursor-pointer transition-colors duration-300">
          Detection Results ( {details.length} )
        </summary>
        <div className="transition-all duration-300 ease-in-out transform origin-top group-open:animate-details-show">
          <p className="text-center text-gray-600 py-2">No object detected</p>
        </div>
      </details>
    );
  }

  return (
    <details className="text-gray-600 group px-2">
      <summary className="my-3 hover:text-gray-600 cursor-pointer transition-colors duration-300">
        Detection Results ( {details.length} )
      </summary>
      <div
        className="transition-all duration-300 ease-in-out transform origin-top
                group-open:animate-details-show"
      >
        <table
          className="text-left responsive-table mx-auto border-collapse table-auto text-sm 
              bg-gray-800 rounded-md overflow-hidden"
        >
          <thead className="bg-gray-700">
            <tr>
              <th className="border-b border-gray-600 p-2 md:p-4 text-gray-100">
                ID
              </th>
              <th className="border-b border-gray-600 p-2 md:p-4 text-gray-100">
                ClassName
              </th>
              <th className="border-b border-gray-600 p-2 md:p-4 text-gray-100">
                Confidence
              </th>
            </tr>
          </thead>
          <tbody>
            {details.map((item, index) => (
              <tr
                key={index}
                className="hover:bg-gray-700 transition-colors text-gray-300"
              >
                <td className="border-b border-gray-600 p-2 md:p-4">
                  {item.track_id}
                </td>
                <td className="border-b border-gray-600 p-2 md:p-4">
                  {classes.class[item.cls_idx]}
                </td>
                <td className="border-b border-gray-600 p-2 md:p-4">
                  {(item.score * 100).toFixed(1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}

function App() {
  const [modelState, setModelState] = useState({
    warnUpTime: 0,
    inferenceTime: 0,
    statusMsg: "Model not loaded",
    statusColor: "inherit",
  });
  const { warnUpTime, inferenceTime, statusMsg, statusColor } = modelState;

  // resource reference
  const backendSelectorRef = useRef(null);
  const modelSelectorRef = useRef(null);
  const cameraSelectorRef = useRef(null);
  const imgszTypeSelectorRef = useRef(null);
  const sessionRef = useRef(null);
  const modelConfigRef = useRef(null);
  const modelCache = useRef({});

  // content reference
  const imgRef = useRef(null);
  const overlayRef = useRef(null);
  const cameraRef = useRef(null);
  const fileImageRef = useRef(null);
  const fileVideoRef = useRef(null);

  const videoProcessingActiveRef = useRef(false);

const updateImgszTypeLock = (modelValue = null) => {
  const selectEl = imgszTypeSelectorRef.current;
  if (!selectEl) return;

  const selected = modelValue || modelSelectorRef.current?.value;
  const isV8 = selected === "yolov8s" || selected === "yolov8n";

  if (isV8) {
    if (selectEl.value !== "zeroPad") {
      selectEl.value = "zeroPad";
    }
    selectEl.disabled = true;
  } else {
    selectEl.disabled = false;
    // Reset value when unlocked to avoid stuck at zeroPad
    if (selectEl.value === "zeroPad") {
      selectEl.value = "dynamic";
    }
  }
};

  // state
  const [customModels, setCustomModels] = useState([]);
  const [cameras, setCameras] = useState([]);
  const [imgSrc, setImgSrc] = useState(null);
  const [details, setDetails] = useState([]);
  const [activeFeature, setActiveFeature] = useState(null); // null, 'video', 'image', 'camera'

  // worker
  //const videoWorkerRef = useRef(null);

  const videoRef = useRef(null);

  const activeFeatureRef = useRef(activeFeature);

  useEffect(() => {
    activeFeatureRef.current = activeFeature;
  }, [activeFeature]);

  // Init page
  useEffect(() => {
    // Delay the lock until DOM is ready
    setTimeout(() => {
      updateImgszTypeLock(modelSelectorRef.current?.value);
    }, 0);
    loadModel();
    getCameras();

    // videoWorkerRef.current = new Worker(
    //   new URL("./utils/video_process_worker.js", import.meta.url),
    //   {
    //     type: "module",
    //   }
    // );
    // videoWorkerRef.current.onmessage = (e) => {
    //   setModelState((prev) => ({
    //     ...prev,
    //     statusMsg: e.data.statusMsg,
    //   }));
    //   if (e.data.processedVideo) {
    //     const url = URL.createObjectURL(e.data.processedVideo);
    //     const a = document.createElement("a");
    //     a.href = url;
    //     a.download = "processed_video.mp4";
    //     a.click();
    //     URL.revokeObjectURL(url);
    //     setActiveFeature(null);
    //   }
    // };
  }, []);

  const loadModel = useCallback(async () => {
    // Always lock immediately using the ACTUAL selected model
    updateImgszTypeLock(modelSelectorRef.current?.value);
    // update model state
    setModelState((prev) => ({
      ...prev,
      statusMsg: "Loading model...",
      statusColor: "red",
    }));

    setActiveFeature("loading");

    // get model config
    const backend = backendSelectorRef.current?.value || "webgpu";
    const selectedModel = modelSelectorRef.current?.value || "yolov8n";

    const customModel = customModels.find(
      (model) => model.url === selectedModel
    );

    const model_path = customModel
      ? customModel.url
      : `${window.location.href}/models/${selectedModel}.onnx`;

    modelConfigRef.current = { model_path, backend };

    const cacheKey = `${selectedModel}-${backend}`;
    if (modelCache.current[cacheKey]) {
      sessionRef.current = modelCache.current[cacheKey];
      setModelState((prev) => ({
        ...prev,
        statusMsg: "Model loaded from cache",
        statusColor: "green",
      }));
      setActiveFeature(null);
      return;
    }

    try {
      // load model
      const start = performance.now();
      const yolo_model = await model_loader(model_path, backend);
      const end = performance.now();

      sessionRef.current = yolo_model;
      modelCache.current[cacheKey] = yolo_model;

      setModelState((prev) => ({
        ...prev,
        statusMsg: "Model loaded",
        statusColor: "green",
        warnUpTime: (end - start).toFixed(2),
      }));
      setTimeout(() => {
      updateImgszTypeLock(modelSelectorRef.current?.value);
      }, 5);
    } catch (error) {
      setModelState((prev) => ({
        ...prev,
        statusMsg: "Model loading failed",
        statusColor: "red",
      }));
      console.error(error);
    } finally {
      setActiveFeature(null);
      setTimeout(() => {
      updateImgszTypeLock(modelSelectorRef.current?.value);
    }, 5);
    }
  }, [customModels]);

  const getCameras = useCallback(async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(
        (device) => device.kind === "videoinput"
      );
      setCameras(videoDevices);
    } catch (err) {
      console.error("Error getting cameras:", err);
    }
  }, []);

  const handle_AddModel = useCallback((event) => {
    const file = event.target.files[0];
    if (file) {
      const fileName = file.name.replace(".onnx", "");
      const fileUrl = URL.createObjectURL(file);
      setCustomModels((prevModels) => [
        ...prevModels,
        { name: fileName, url: fileUrl },
      ]);
    }
  }, []);

  const handle_OpenImage = useCallback(
    (imgUrl = null) => {
      if (imgUrl) {
        setImgSrc(imgUrl);
        setActiveFeature("image");
      } else if (imgSrc) {
        if (imgSrc.startsWith("blob:")) {
          URL.revokeObjectURL(imgSrc);
        }
        overlayRef.current.width = 0;
        overlayRef.current.height = 0;
        setImgSrc(null);
        setDetails([]);
        setActiveFeature(null);
        setTimeout(() => {
          updateImgszTypeLock(modelSelectorRef.current?.value);
        }, 5);
      }
    },
    [imgSrc]
  );

  const handle_ImageLoad = useCallback(async () => {
    overlayRef.current.width = imgRef.current.width;
    overlayRef.current.height = imgRef.current.height;

    try {
      const src_mat = cv.imread(imgRef.current);
      const [results, results_inferenceTime] = await inference_pipeline(
        src_mat,
        sessionRef.current,
        new BYTETracker(),
        [overlayRef.current.width, overlayRef.current.height],
        imgszTypeSelectorRef.current.value
      );
      const overlayCtx = overlayRef.current.getContext("2d");
      overlayCtx.clearRect(
        0,
        0,
        overlayCtx.canvas.width,
        overlayCtx.canvas.height
      );

      draw_bounding_boxes(results, overlayCtx);
      setDetails(results);
      setModelState((prev) => ({
        ...prev,
        inferenceTime: results_inferenceTime,
      }));
    } catch (error) {
      console.error("Image processing error:", error);
    }
  }, [sessionRef.current]);

  const handle_CloseVideo = useCallback(() => {
    if (!videoRef.current) return;

    videoProcessingActiveRef.current = false;

    // Stop the video playback and clear src
    videoRef.current.pause();
    videoRef.current.removeAttribute("src");
    videoRef.current.load();

    // Clear overlay canvas
    if (overlayRef.current) {
      const ctx = overlayRef.current.getContext("2d");
      ctx.clearRect(0, 0, overlayRef.current.width, overlayRef.current.height);
      overlayRef.current.width = 0;
      overlayRef.current.height = 0;
    }

    // Clear detection details
    setDetails([]);

    // Clear active feature
    setActiveFeature(null);
    setTimeout(() => {
      updateImgszTypeLock(modelSelectorRef.current?.value);
    }, 5);
  }, []);

  const handle_ToggleCamera = useCallback(async () => {
    if (cameraRef.current.srcObject) {
      // stop camera
      cameraRef.current.srcObject.getTracks().forEach((track) => track.stop());
      cameraRef.current.srcObject = null;
      overlayRef.current.width = 0;
      overlayRef.current.height = 0;

      setDetails([]);
      setActiveFeature(null);
      setTimeout(() => {
      updateImgszTypeLock(modelSelectorRef.current?.value);
    }, 5);
    } else if (cameraSelectorRef.current && cameraSelectorRef.current.value) {
      try {
        // open camera
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            deviceId: cameraSelectorRef.current.value,
          },
          audio: false,
        });
        cameraRef.current.srcObject = stream;
        setActiveFeature("camera");
      } catch (err) {
        console.error("Error accessing camera:", err);
      }
    }
  }, []);

  const handle_cameraLoad = useCallback(() => {
    overlayRef.current.width = cameraRef.current.clientWidth;
    overlayRef.current.height = cameraRef.current.clientHeight;

    let inputCanvas = new OffscreenCanvas(
      cameraRef.current.videoWidth,
      cameraRef.current.videoHeight
    );
    let ctx = inputCanvas.getContext("2d", {
      willReadFrequently: true,
    });
    const tracker = new BYTETracker();

    const handle_frame_continuous = async () => {
      if (!cameraRef.current?.srcObject) {
        inputCanvas = null;
        ctx = null;
        return;
      }
      ctx.drawImage(
        cameraRef.current,
        0,
        0,
        cameraRef.current.videoWidth,
        cameraRef.current.videoHeight
      ); // draw camera frame to input canvas
      const src_mat = cv.imread(inputCanvas);
      const [results, results_inferenceTime] = await inference_pipeline(
        src_mat,
        sessionRef.current,
        tracker,
        [overlayRef.current.width, overlayRef.current.height],
        imgszTypeSelectorRef.current.value
      );
      const overlayCtx = overlayRef.current.getContext("2d");
      overlayCtx.clearRect(
        0,
        0,
        overlayCtx.canvas.width,
        overlayCtx.canvas.height
      );
      draw_bounding_boxes(results, overlayCtx);

      setDetails(results);
      setModelState((prev) => ({
        ...prev,
        inferenceTime: results_inferenceTime,
      }));

      requestAnimationFrame(handle_frame_continuous);
    };
    requestAnimationFrame(handle_frame_continuous);
  }, [sessionRef.current]);

const handle_OpenVideo = useCallback((file) => {
  if (!file) {
    setActiveFeature(null);
    return;
  }
  videoProcessingActiveRef.current = true;
  let isProcessing = false;

  const url = URL.createObjectURL(file);
  videoRef.current.src = url;

videoRef.current.onloadedmetadata = () => {
  console.log("Video metadata loaded");

  setActiveFeature("video");

  requestAnimationFrame(() => {
    const displayWidth = videoRef.current.clientWidth || videoRef.current.videoWidth;
    const displayHeight = videoRef.current.clientHeight || videoRef.current.videoHeight;

    console.log("Video displayed size:", displayWidth, displayHeight);

    overlayRef.current.width = displayWidth;
    overlayRef.current.height = displayHeight;

    let inputCanvas = new OffscreenCanvas(videoRef.current.videoWidth, videoRef.current.videoHeight);
    let ctx = inputCanvas.getContext("2d", { willReadFrequently: true });
    const tracker = new BYTETracker();

    // Make processFrame async so we can await inference and only then schedule next frame
    const processFrame = async (now, metadata) => {
      if (!videoProcessingActiveRef.current || activeFeatureRef.current !== "video") return;

      if (isProcessing) {
        // Skip this frame if inference is busy to avoid overlap
        videoRef.current.requestVideoFrameCallback(processFrame);
        return;
      }
      isProcessing = true;

      try {

      ctx.drawImage(videoRef.current, 0, 0, videoRef.current.videoWidth, videoRef.current.videoHeight);

      const src_mat = cv.imread(inputCanvas);

      const [results, inferTime] = await inference_pipeline(
        src_mat,
        sessionRef.current,
        tracker,
        [overlayRef.current.width, overlayRef.current.height],
        imgszTypeSelectorRef.current.value
      );

      const overlayCtx = overlayRef.current.getContext("2d");
      overlayCtx.clearRect(0, 0, overlayCtx.canvas.width, overlayCtx.canvas.height);
      draw_bounding_boxes(results, overlayCtx);

      setDetails(results);
      setModelState(prev => ({ ...prev, inferenceTime: inferTime }));

      } catch (error) {
    console.error("Inference error:", error);
  } finally {
    isProcessing = false;
    // Schedule next frame only if still active
    if (videoProcessingActiveRef.current && activeFeatureRef.current === "video") {
      videoRef.current.requestVideoFrameCallback(processFrame);
    }
  }
};

// Start frame processing synced with video frame rendering
  videoRef.current.requestVideoFrameCallback(processFrame);
});

};

}, [sessionRef.current, activeFeature]);


  return (
    <>
      <h2 className="my-6 md:my-4 text-3xl md:text-4xl px-2" style={{fontSize: "24px", color: "#f77a06ff"}}>📟 Makers - YOLOv11 ByteTrack Object Tracking </h2>

      <center className="container text-lg w-full md:text-xl px-2 mb-2" style={{borderRadius: "8px", border: "2px solid #2192a1ff"}}>
      <Marquee direction={"left"} speed={50} gradient={false} style={{ fontSize: "16px", color: "#11beddff", fontWeight: "bold"}}>
        Multi Object Tracking on Web — COCO-80 Classes — YOLOv11 & YOLOv8 in ONNX.&nbsp;&nbsp;
      </Marquee>
      </center>

      <SettingsPanel
        backendSelectorRef={backendSelectorRef}
        modelSelectorRef={modelSelectorRef}
        cameraSelectorRef={cameraSelectorRef}
        imgszTypeSelectorRef={imgszTypeSelectorRef}
        cameras={cameras}
        customModels={customModels}
        onModelChange={loadModel}
        activeFeature={activeFeature}
        updateImgszTypeLock={updateImgszTypeLock}
      />

      <ImageDisplay
        cameraRef={cameraRef}
        videoRef={videoRef}   // pass videoRef here
        imgRef={imgRef}
        overlayRef={overlayRef}
        imgSrc={imgSrc}
        onCameraLoad={handle_cameraLoad}
        onImageLoad={handle_ImageLoad}
        activeFeature={activeFeature}
      />

      <ControlButtons
        cameras={cameras}
        imgSrc={imgSrc}
        fileVideoRef={fileVideoRef}
        fileImageRef={fileImageRef}
        handle_OpenVideo={handle_OpenVideo}
        handle_CloseVideo={handle_CloseVideo}
        handle_OpenImage={handle_OpenImage}
        handle_ToggleCamera={handle_ToggleCamera}
        handle_AddModel={handle_AddModel}
        activeFeature={activeFeature}
      />

      <ModelStatus
        warnUpTime={warnUpTime}
        inferenceTime={inferenceTime}
        statusMsg={statusMsg}
        statusColor={statusColor}
      />

      <ResultsTable details={details} />
    </>
  );
}

export default App;
