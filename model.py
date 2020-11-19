# coding: utf-8
# 2020/인공지능/final/학번/이름
import sys
import os
from collections import OrderedDict
import pickle
import numpy as np
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)

        return y.T

    x = x - np.max(x)  # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask=(x<=0)
        out=x.copy()
        out[self.mask]=0

        return out

    def backward(self, dout):
        dout[self.mask]=0
        dx=dout

        return dx


class CustomActivation:
    # 시그모이드 함수를 활성화 함수로 사용 
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    #Affine 계층 구현
    def __init__(self, W, b):
        self.W=W
        self.b=b
        self.x=None
        self.dw=None
        self.db=None

    def forward(self, x):
        self.x=x
        out=np.dot(x,self.W)+self.b

        return out

    def backward(self, dout):
        dx=np.dot(dout,self.W.T)
        self.dw=np.dot(self.x.T,dout)
        self.db=np.sum(dout,axis=0)

        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss=None
        self.y=None
        self.t=None

    def forward(self, x, t):
        self.t=t
        self.y=softmax(x)
        self.loss=cross_entropy_error(self.y,self.t)
        

        return self.loss

    def backward(self, dout=1):
        batch_size=self.t.shape[0]
        dx=(self.y-self.t)/batch_size

        return dx


class SGD:
    #SGD를 사용하지않을 예정
    def __init__(self, lr=0.0001):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            if key != 'std':
                if key != 'mean':
                    params[key]-=self.lr *grads[key]


class CustomOptimizer:
    # 밑바닥부터 시작하는 딥러닝에 나오는 Adam이라는 Optimizer 사용
    # 아담은 모멘텀과 에이다그레이드 를 합친 방법 
    def __init__(self, lr=0.001, beta1=0.9 , beta2=0.999):
            self.lr = lr
            self.beta1 = beta1
            self.beta2 = beta2
            self.iter = 0
            self.m = None
            self.v = None
        
    def update(self, params, grads):
        '''
        수도코드를 확인해보면 아담은 1차모멘텀과 2차모멘텀을 사용하는 방식이다.
        1차 모멘텀은 m이고 2차모멘텀은 v이다. 
        '''
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                if key != 'std' :
                    if key != 'mean':
                        # m과 v의 값을 기준으로 계산하기 때문에 이 값들을 초기화한다.
                        self.m[key] = np.zeros_like(val)
                        self.v[key] = np.zeros_like(val)
        #업데이트 진행시마다 iter 변수의 값 늘려 점점 lr_t가 작아지게함 
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        #m^t 와 v^t 에 관한 식을 
        for key in params.keys():
            #각각의 파라미터에 대해서 작동한다. 
            if key != 'std' :
                if key != 'mean':
                    '''
                    m1 = beta * m0 + (1-beta) *  g 이다 .
                    이렇게 함으로써 m0가 0일때 학습초반에 0으로 bias되는 문제를 해결했다. 
                    '''
                    self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
                    # self.m[key]= self.m[key] +  (g - self.m[key]) - b1* g + b1 * self.m[key]
                    #  이 식을 다르게 쓰면
                    #1번식  self.m[key]= self.beta1 * self.m[key] + (1-self.beta1) grads[key] 로 쓸수있다.
                    
                    self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
                    # 이식도 마찬가지로  풀어서 식을 변형하면
                    # self.v[key]= self.v[key] + grad ^2 - self.v[key] - grad^2 * b2 + v*b2 이므로
                    #2번식  self.v[key]= self.beta2 * self.v[key] (1- self.beta2) grads[key]**2 로 쓸수있다.
                    # 1번식과 2번식은 adam 의 수도코드를 그대로 이용한 식이다. 추후 보고서 서술
                    params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            


class Model:
    """
    네트워크 모델 입니다.

    """
    def __init__(self, lr=0.001,l_num=4):
        """
        클래스 초기화
        """
        
        self.l_num= l_num
        self.params = {}
        self.__init_weight(6,6)
        self.__init_layer()
        self.optimizer = CustomOptimizer(lr)
        
    def __init_layer(self):
        """
        레이어를 생성하시면 됩니다.
        """
        self.layers=OrderedDict()
        '''
        히든 레이어의 갯수는 l_num -1 개이다. 
        '''
        for i in range(1,self.l_num):
            self.layers["Affine%d" % i]=Affine(self.params['W%d'% i],self.params['b%d'% i])
            if i%2==0:
                self.layers['custom%d'%i]=CustomActivation()
            else :
                self.layers['custom%d'%i]=CustomActivation()
        # 레이어를 나누어서 홀수일때는 ReLu , 짝수일 때는 sigmoid 를 사용하기 위한 히든레이어 생성 코드
        
        self.layers['Affine%d'% (self.l_num)]= Affine(self.params['W%d'% (self.l_num)],self.params['b%d'% (self.l_num)])
        self.last_layer=SoftmaxWithLoss()
            
    def __init_weight(self,input_size,output_size):
        """
        레이어에 탑재 될 파라미터들을 초기화 하시면 됩니다.
        """
        self.params={}
        test=50
        xavier=np.sqrt(test)
       
        # xavier 초기값을 사용한다. test는 히든레이어의 노드수이다.
        self.params['W1']= np.random.randn(input_size,test)/ np.sqrt(6)
        # input feature 가 6개이므로 첫 히든레이어의 초기값은 6이다.
        self.params['b1']=np.zeros(test)
        for i in range(2,self.l_num):
       
            self.params['W'+str(i)]= np.random.randn(test,test)/ xavier
            self.params['b'+str(i)]=np.zeros(test)

        self.params['W'+str(self.l_num)]=np.random.randn(test,output_size)/xavier
        self.params['b'+str(self.l_num)]=np.zeros(output_size)
        
    def scaling(self,x):
        # 스케일링을 위한 함수를 설정하였다. 
        self.std=self.params['std']
        self.m=self.params['mean']
        x_scale=(x-self.m)/self.std
        return x_scale
    def update(self, x, t,std=np.zeros(6),m=np.zeros(6)):
        """
        train 데이터와 레이블을 사용해서 그라디언트를 구한 뒤
         옵티마이저 클래스를 사용해서 네트워크 파라미터를 업데이트 해주는 함수입니다.

        :param x: train_data
        :param t: test_data
        """
        
        self.params['std']=std
        self.params['mean']=m
        # 스케일링을 위한 전체 train data의 평균과 표준편차를 저장한다.
        grads = self.gradient(x, t)
        self.optimizer.update(self.params, grads)

    def predict(self, x):
        """
        데이터를 입력받아 정답을 예측하는 함수입니다.

        :param x: data
        :return: predicted answer
        """
        x=self.scaling(x)
        # ㅇ
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        """
        데이터와 레이블을 입력받아 로스를 구하는 함수입니다.
        :param x: data
        :param t: data_label
        :return: loss
        """
        
        y = self.predict(x)
        return self.last_layer.forward(y, t)


    def gradient(self, x, t):
        """
        train 데이터와 레이블을 사용해서 그라디언트를 구하는 함수입니다.
        첫번째로 받은데이터를 forward propagation 시키고,
        두번째로 back propagation 시켜 grads에 미분값을 리턴합니다.
        :param x: data
        :param t: data_label
        :return: grads
        """
        #x=self.scaling(x)
         #forward
        self.loss(x,t)
        # backward
        dout=1
        dout=self.last_layer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout= layer.backward(dout)
        # 결과 저장
        grads = {}
        for i in range(1,self.l_num+1):
            grads['W%d' % i] = self.layers['Affine%d'% i].dw
            grads['b%d' % i] = self.layers['Affine%d'% i].db
        

        return grads

    def save_params(self, file_name="params.pkl"):
        """
        네트워크 파라미터를 피클 파일로 저장하는 함수입니다.

        :param file_name: 파라미터를 저장할 파일 이름입니다. 기본값은 "params.pkl" 입니다.
        """
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        """
        저장된 파라미터를 읽어와 네트워크에 탑재하는 함수입니다.

        :param file_name: 파라미터를 로드할 파일 이름입니다. 기본값은 "params.pkl" 입니다.
        """
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val
        self.__init_layer()
        pass
