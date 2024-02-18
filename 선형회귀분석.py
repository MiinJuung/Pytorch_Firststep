import torch

'''
1. 텐서 생성하기
'''
X = torch.Tensor(2,3) # random 하게 2*3 텐서 생성
X = torch.Tensor([[1,2,3],[4,5,6]])  # 정해진 값으로 2*3 텐서 생성
X = torch.IntTensor(2,3) # 다양한 자료형으로 저장할 수 있도록 함
print(X)

'''
2. 생성한 텐서를 이용해서 오차에 대한 경사하강법을 적용한 연산 그래프 보기
'''
x = torch.tensor([2.0,3.0],requires_grad=True)
print(x)
y = x**2
z = 2*y +3

target = torch.Tensor([3.0,4.0])
loss = torch.sum(torch.abs(z-target))
loss.backward()   # 쉽게 미분 시켜줌

print(x.grad,y.grad,z.grad)

'''
3. 선형회귀모델 만들어서 기울기 계산 및 w,b 업데이트 과정 확인
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

num_data = 1000
num_epoch = 500

x = init.uniform_(torch.Tensor(num_data,1),-10,10)
noise = init.normal_(torch.FloatTensor(num_data,1),std = 1)
y = 2*x + 3
y_noise = 2*(x+noise) + 3

########## 시각화 ###########
import matplotlib.pyplot as plt
import pandas as pd

plt.figure(figsize=(12,8))
plt.scatter(x, y_noise, alpha=0.2)
plt.show()
##############################

model = nn.Linear(1,1)  # 1개의 특성을 가진 X, 1개의 특성을 가진 Y 를 인수로 가지는 선형회귀모델
loss_func = nn.L1Loss()  # L1loss 사용
optimizer = optim.SGD(model.parameters(), lr = 0.01) # 최적화 함수 지정


########## 선형 모형 학습 결과 시각화 ###########
from tqdm.notebook import tqdm

label = y_noise
loss_lst = []

fig = plt.figure(figsize=(24,16))

for n in tqdm(range(num_epoch)):
    optimizer.zero_grad() # 기울기를 0으로 초기화
    output = model(x)
    
    loss = loss_func(output,label)
    loss.backward()
    optimizer.step() # 변수들의 기울기 업데이트
    
    if n % 50 == 0:
#         print(loss.data)
        
        plt.subplot(3, 4, n//50 + 1)
        plt.scatter(x, y_noise, alpha=0.2)
        plt.plot([-10, 10], model(torch.Tensor([[-10],[10]])).squeeze().detach().numpy(), 
                 color='r', alpha=(n/num_epoch))
        
        for i in range(num_data):
            y_true = y_noise[i].item()
            y_pred = model(torch.Tensor([x[i]])).detach().numpy()
            plt.vlines(x[i],min(y_true,y_pred), max(y_true, y_pred), color='red', alpha=0.01)
        
        plt.title(f"Epoch {n} Loss : {loss.item()}")
        plt.grid(True)
    
    loss_lst.append(loss.data.item())

plt.show()

param_list = list(model.parameters())
print(param_list[0].item(), param_list[1].item())


##################### loss 줄어드는 것 시각화 #########################
loss_df = pd.DataFrame(loss_lst)
loss_df.plot(figsize=(8,6))
plt.legend(["train loss"])
plt.grid(True)
plt.show()

############################
# label = y_noise
# for i in range(num_epoch):   #epoch만큼 학습
#     optimizer.zero_grad()    # 각 반복시 기울기를 0으로 계속 초기화
#     output = model(x)      # 정의한 model output 저장    

#     loss = loss_func(output,label)   # loss 함수 지정
#     loss.backward()    # 각 변수 w,b에 대한 기울기 계산
#     optimizer.step()   

#     if i%10 == 0:
#         print(loss.data)
#     param_list = list(model.parameters())       # optimizer.step()를 호출하여 인수로 들어갔던 model.parameters() 에서 리턴되는 변수들의 기울기에 학습률 0.01을 곱하여 빼줌으로써 업데이트
#     print(param_list[0].item(),param_list[1].item()) # 처음에 의도한 것 처럼 2와 3이라는 가중치와 편차에 가까운 값 나옴.


