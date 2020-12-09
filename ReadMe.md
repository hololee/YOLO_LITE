## YOLO LITE - pytorch

#### YOLO-LITE
This project based on below paper.  
The model structure on this paper is modified for my own dataset.  
   
`YOLO-LITE: A Real-Time Object Detection Algorithm Optimized for Non-GPU Computers`

  
 

#### history
yololite 논문에서는작은 네트워크 사이즈로 인해서 BN 의 사용 없이도 충분히 학습가능하다 언급.  
dataset 의 특성으로 인해서인지 학습이 잘 되지 않는 현상을 발견.  
conv layer를 추가 해주고 BN layer를 추가.

prediction 이 안좋음 (weed 를 중구 난방으로 찾아냄)
- weed 를 제외한 bean만 찾도록 데이터를 재구성.
