## YOLO LITE - pytorch

### YOLO LITE
This project is based on below paper.  
The model structure on this paper is modified for my own dataset.  
   
[`YOLO-LITE: A Real-Time Object Detection Algorithm Optimized for Non-GPU Computers`](https://arxiv.org/pdf/1811.05588)

---  

### Abstract  
YOLO는 오브젝트 디텍션 분야에서 좋은 성적을 보여주었으며 R-CNN 페밀리와 비교 하였을때 구현이 간단하고 모델이 더 가볍다는 특징이 있다.
YOLO lite는 YOLO를 이용하여 최대한 모델을 작게 구성하여 로컬 환경에서 CPU만을 이용하여 사용가능한 뉴럴네트워크 모델을 제안하였다.
여기서는 YOLO lite 모델을 좀더 개선하여 밭에서 자라나는 콩 작물을 검출하는 농기계에서 사용 가능하도록 구성하였다.
  
### Introduction  
뉴럴 네트워크는 GPU의 발전과 함께 급속도로 발전하였다.  
좀더 많은 연산을 빠르게 처리하면서 퍼포먼스는 증가하고 실생활에서 사람들보다 더 좋은 성능을 보여주기도 하였다.
컴퓨터 비전의 오브젝트 디텍션 문제는 뉴럴네트워크를 도입 하면서 급속도의 성장을 보여주었다.
물체 인식, 얼굴인식, 상품 품질 체크와 같은 일반적인 detection을 활용한 분야뿐 아니라 cctv를 이용한 사람 이동 추적,  자율주행 자동차와 같은 여러 응용 분야에도  적용되고 있다. 
현 시점에서도 GPU의 발전은 가속화 되고 있고, 그에 따라서 학습 모델은 더욱더 무거워지고  복잡해지고 있다.
하지만 여전히 많은 상황들, 자율주행 자동차, cctv 등은 적은 자원으로 동작하기 때문에 GPU의 지원 한계가 있다.   
이 프로젝트 에서는 이러한 GPU를 활용하기 어려운 환경에서 CPU만을 사용하여 뉴럴네트워크를 동작시켜 보고자 한다.
  
### Dataset  
목표는 밭에서 기르는 콩 작물에 대한 객체 검출이다. 검출된 영역은  작물의 상태나 성장 상태의 파악을 위해서 쓰일 수 있다.  
  
img1|img2|img3|img4
:----:|:----:|:----:|:----:
![img](https://raw.githubusercontent.com/hololee/YOLO_LITE/master/data/bean_leaf/images/a1.png)|![img](https://raw.githubusercontent.com/hololee/YOLO_LITE/master/data/bean_leaf/images/a2.png)|![img](https://raw.githubusercontent.com/hololee/YOLO_LITE/master/data/bean_leaf/images/a3.png)|![img](https://raw.githubusercontent.com/hololee/YOLO_LITE/master/data/bean_leaf/images/a4.png)


  
### Model   
여기서는 YOLOv1 을 기반으로 하는 YOLO-lite를 이용하였다. CPU 라는 제한적인 환경에서 운영되지만  성능 향상을 위해서 Batch normalization 을 추가하였고 기존의 YOLO lite 에서 한층의 conv filter를 사용한것과는 다르게 2개의 층을 이용하였다.
아래의 사진은 model 구조를 보여준다.  
![model](https://raw.githubusercontent.com/hololee/YOLO_LITE/master/models/model.jpg)  
  

아래의 내용은 model 을 학습 시킴에 있어서 하이퍼 파라미터를 보여준다.

~~~
-- public parpams --
Dataset devide ratio : [0.8 ,0.1, 0.1]
Learning rate : 0.0001
Batch size : 16
Epoch : 300

-- yolo params --
Lambda_coord = 5
Lambda_noobj = 0.5
Grid = 8 x 8

-- post process params --
OUTPUT_THRESHOLD = 0.5
NMS_THRESHOLD = 0.5
~~~

  
### Experiment
학습은 아래와 같은 조건 하에서 이루어졌다.      

Type | Condition
:----:|:----:
A | with weed, 2box prediction
B | with weed, 1box prediction  
C | without weed, 2box prediction  
D | without weed, 1box prediction  

아래의 그래프는 각각의 조건에 따른 학습 그래프를 보여주고 있다.

성능 평가는 mAP 를 이용하였으며 아래 테이블은 그 결과 이다.  
- 일반적으로 class에 대한 평균 AP 를 mAP라 하지만 with weed 와 비교하기 위해 mAP로 통일한다.  

성능 평가를 제외한 NMS 이후의 출력을 이용하여 FPS 를 계산한 결과는 아래와 같다.
- stochastic 방법으로 한번에 하나씩 출력하였다.
- 아래 테이블의 모든 데이터는 CPU 에서 테스트한 결과이다.
  
#### Test Result
Type|mAP|AP<sub>bean</sub>|AP<sub>weed</sub>|FPS|FPS<sub>w/o NMS</sub>
:----:|:----:|:----:|:----:|:----:|:----:
A |**0.6156**|0.6811|0.5501|3.59|4.86
B |0.5579|0.6200|0.4959|4.29|4.71
C | **0.7240**|-|-|3.90|4.72|
D |0.7195|-|-|4.24|4.69|
  
### Result 

- #### leanring graph
    TYPE|graph| 
    :----:|:----:
    A |![A_learning](https://raw.githubusercontent.com/hololee/YOLO_LITE/master/output/bean_leaf/learning_graph(weed%2C%202boxes).png)
    B |![B_learning](https://raw.githubusercontent.com/hololee/YOLO_LITE/master/output/bean_leaf/learning_graph(weed%2C%201boxes).png)
    C |![C_learning](https://raw.githubusercontent.com/hololee/YOLO_LITE/master/output/bean_leaf_noweed/learning_graph(noweed%2C%202boxes).png)
    D |![D_learning](https://raw.githubusercontent.com/hololee/YOLO_LITE/master/output/bean_leaf_noweed/learning_graph(noweed%2C%201boxes).png)  
  
  
- #### Precision-Recall curve
    TYPE|Bean|Weed 
    :----:|:----:|:----:
    A |![A_PR_curve](https://raw.githubusercontent.com/hololee/YOLO_LITE/master/output/bean_leaf/precision_recall_curve_Bean(weed%2C%202boxes).png)|![A_learning](https://raw.githubusercontent.com/hololee/YOLO_LITE/master/output/bean_leaf/precision_recall_curve_Weed(weed%2C%202boxes).png)
    B |![B_PR_curve](https://raw.githubusercontent.com/hololee/YOLO_LITE/master/output/bean_leaf/precision_recall_curve_Bean(weed%2C%201boxes).png)|![A_learning](https://raw.githubusercontent.com/hololee/YOLO_LITE/master/output/bean_leaf/precision_recall_curve_Weed(weed%2C%201boxes).png)
    C |![C_PR_curve](https://raw.githubusercontent.com/hololee/YOLO_LITE/master/output/bean_leaf_noweed/precision_recall_curve(noweed%2C%202boxes).png)|-
    D |![D_PR_curve](https://raw.githubusercontent.com/hololee/YOLO_LITE/master/output/bean_leaf_noweed/precision_recall_curve(noweed%2C%201boxes).png)|-
  
  
- #### Output Result
    Type|sample1|sample2|sample3
    :----:|:----:|:----:|:----:
    A|![output](https://raw.githubusercontent.com/hololee/YOLO_LITE/master/output/A_2.png)|![output](https://raw.githubusercontent.com/hololee/YOLO_LITE/master/output/A_3.png)|![output](https://raw.githubusercontent.com/hololee/YOLO_LITE/master/output/A_4.png)
    B|![output](https://raw.githubusercontent.com/hololee/YOLO_LITE/master/output/B_2.png)|![output](https://raw.githubusercontent.com/hololee/YOLO_LITE/master/output/B_3.png)|![output](https://raw.githubusercontent.com/hololee/YOLO_LITE/master/output/B_4.png)
    C|![output](https://raw.githubusercontent.com/hololee/YOLO_LITE/master/output/C_2.png)|![output](https://raw.githubusercontent.com/hololee/YOLO_LITE/master/output/C_3.png)|![output](https://raw.githubusercontent.com/hololee/YOLO_LITE/master/output/C_4.png)
    D|![output](https://raw.githubusercontent.com/hololee/YOLO_LITE/master/output/D_2.png)|![output](https://raw.githubusercontent.com/hololee/YOLO_LITE/master/output/D_3.png)|![output](https://raw.githubusercontent.com/hololee/YOLO_LITE/master/output/D_4.png)
      

#### Next step.
위의 실험은 실수로 backward() 수행 전 optimizer의 zero_grad()를 수행하지 않은 결과이다.  
따라서 backprop 이 누적 되어서 학습이 되었다.(RNN 과 같은 모델에서 [효율적으로 사용](https://newsight.tistory.com/94))  
zero_grad()를 추가하고 다시 학습 시킨 결과의 그래프는 아래와 같이 generalize가 잘 안되는 결과를 보여주고 있다.  
축적된 grad 값들이 일종의 규제 처럼 작용하여 overfit을 방지 하고 있는 현상으로 보인다.  

참고([Why do we need to set the gradients manually to zero in pytorch?](https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/20))  

![adjust zero_grad](https://raw.githubusercontent.com/hololee/YOLO_LITE/master/output/final.png)

