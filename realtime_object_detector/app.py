import time
import edgeiq

def main():
    #detects the object
    obj_detect = edgeiq.ObjectDetection(
            "alwaysai/mobilenet_ssd")
    obj_detect.load(engine=edgeiq.Engine.DNN)

    print("Loaded model:\n{}\n".format(obj_detect.model_id))
    print("Engine: {}".format(obj_detect.engine))
    print("Accelerator: {}\n".format(obj_detect.accelerator))
    print("Labels:\n{}\n".format(obj_detect.labels))

    fps = edgeiq.FPS()

    try:
        with edgeiq.WebcamVideoStream(cam=0) as video_stream, \
                edgeiq.Streamer() as streamer:
            # Allow Webcam to warm up
            time.sleep(2.0)

            tracker = edgeiq.CentroidTracker(deregister_frames=20, max_distance=50)

            fps.start()
            
            objects = {}
            objectsCopy = {}
            # loop detection
            while True:
                frame = video_stream.read()
                frame = edgeiq.resize(frame, width=600)
                results = obj_detect.detect_objects(frame, confidence_level=.5)
                frame = edgeiq.markup_image(
                        frame, results.predictions, colors=obj_detect.colors)

                # Generate text to display on streamer
                text = ["Model: {}".format(obj_detect.model_id)]
                text.append(
                        "Inference time: {:1.3f} s".format(results.duration))

                text.append("Item Count:")
                objectsCopy = objects.copy()
                objects = tracker.update(results.predictions)

                if len(objects) < len(objectsCopy):
                    for key in objects:
                        del objectsCopy[key]
                    for key in objectsCopy:
                        text.append(("%s has been stolen!" % objectsCopy[key].label).format(results.duration))

                #if len(objects) < count:
                #    print('something left the frame')

                #count = len(objects)

                
                #predictions = []
                #for (object_id, prediction) in objects.items():
                #    new_label = 'Object {}'.format(object_id)
                #    prediction.label = new_label
                #    text.append(new_label)
                #    predictions.append(prediction)

                #for prediction in results.predictions:
                 #   text.append("{}: {:2.2f}%".format(
                  #      prediction.label, prediction.confidence * 100))

                streamer.send_data(frame, text)

                fps.update()

                if streamer.check_exit():
                    break

    finally:
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")


if __name__ == "__main__":
    main()
