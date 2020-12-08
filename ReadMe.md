## YOLO LITE - pytorch


#### history
yololite 논문에서는작은 네트워크 사이즈로 인해서 BN 의 사용 없이도 충분히 학습가능하다 언급
dataset 의 특성으로 인해서인지 학습이 잘 되지 않는 현상을 발견.
conv를 한단계씩 추가 해주고 BN layer 를 추가.
=> 학습이 이루어짐.

prediction 이 안좋음 (weed 를 중구 난방으로 찾아냄)
