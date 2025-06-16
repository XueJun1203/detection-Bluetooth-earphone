# detection-Bluetooth-earphone
Embedded detection Bluetooth earphone YOLO model

Insert `det.model` into the SD card and insert the SD card into K210.

Connect the serial port through CanMV, open and run the `det.py` file through CanMV, burn it to K210, and plug it in to use.

`train_loss.txt` is the loss value during the training process

1.Modify the contents of the labels array to match the labels in the model you trained (which can be found in the `label.txt` file)
`anchors` Use the value from the second line of `anchor.txt`
```
labels = ["蓝牙耳机"] 
anchor = (2.40, 1.98, 3.51, 2.24, 5.02, 2.69, 4.95, 3.58, 7.55, 5.99) 
```
2.Initialize KPU object
Change the path in `kpu. load_kmodel (/sd/det.kmodel ')` to your own model path: `kpu. load_kmodel ('/sd/det.kmodel ')`
```
kpu = KPU()
kpu.load_kmodel('/sd/det.kmodel')
#kpu.load_kmodel(0x300000, 584744)
kpu.init_yolo2(anchor, anchor_num=(int)(len(anchor)/2), img_w=320, img_h=240, net_w=320 , net_h=240 ,layer_w=10 ,layer_h=8, threshold=0.6, nms_value=0.3, classes=len(labels))
```
3.Main function of object detection
```
while(True):
    gc.collect()

    clock.tick()
    img = sensor.snapshot()

    kpu.run_with_output(img)
    dect = kpu.regionlayer_yolo2()

    fps = clock.fps()

    if len(dect) > 0:
        for l in dect :
            a = img.draw_rectangle(l[0],l[1],l[2],l[3],color=(0,255,0))

            info = "%s %.3f" % (labels[l[4]], l[5])
            a = img.draw_string(l[0],l[1],info,color=(255,0,0),scale=2.0)
            print(info)
            del info

    a = img.draw_string(0, 0, "%2.1ffps" %(fps),color=(0,60,255),scale=2.0)
    lcd.display(img)
```
