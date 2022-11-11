# HW3: Classification
## 第一件事就是Debug
我因為我的Folder structure可能跟別人不一樣，導致我下載Code回來不能自己跑，分享一下過程
出現以下錯誤:
```
  File "d:\_Code\ML2022-Spring-HW\HW3\2022ml_hw3_image_classification.py", line 182, in <module>
    loss.backward()
  File "C:\Users\xTaiwan\AppData\Local\Programs\Python\Python39\lib\site-packages\torch\_tensor.py", line 487, in backward
    torch.autograd.backward(
  File "C:\Users\xTaiwan\AppData\Local\Programs\Python\Python39\lib\site-packages\torch\autograd\__init__.py", line 197, in backward
    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling `cublasSgemm( handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc)`
```
好奇怪喔，找了很多網站也找不出這個到底是什麼問題，甚至換了CUDA Driver也不行。

> 當時覺得很煩躁，因為弄了一整天QQ
> 
突發一想，會不會是CUDA在搞事，死馬當活馬醫:
`device = cpu`
接著出現以下錯誤
```
Target -1 is out of bounds.
  File "D:\_Code\ML2022-Spring-HW\HW3\hw3.py", line 188, in <module>
    loss = criterion(logits, labels)
```
天哪，越來越奇怪了，這是什麼問題啊==
開Debugger看Variable，發現`labels: tensor([-1])`
於是我Review一次Code，在`class FoodDataset(Dataset)`中`def __getitem__(self, idx):`
發現沒有正確讀到資料就會回傳`-1`，修改一下`label = int(fname.split("/")[-1].split("\\")[-1].split("_")[0])`中split的規則就正常了，沒有更多錯誤發生。

# The result and my method
## Score
測試了幾個network，Train出來效果最好的是`ResNET152`，如圖。
調參數和加了一些優化都沒有辦法過Strong baseline，之後有空再來嘗試
![](https://i.imgur.com/4RskXQp.png)

## Data Augmentation(Transform)
影像增強，這裡我`test_tfm`和`train_tfm`使用了不同的Transform，具體效果可以直接Google Pytorch的相關文檔。
觀察了一下圖片最小約是256x256，故調整Resize大小。
`train_tfm`這邊我的想法是盡可能弄亂一點
然後兩個都加了Normalize

## Test Time Augmentation
簡單來說，就是多用了`train_tfm`讀出來的資料去預測，最後加總結果取max
比例是`test_tfm`:`train_tfm` 6:4
其中`train_tfm`又分4種 各種比例相等
對了，我的Code真的寫得很醜，鞭小力一點

## Cross Validation(Kfold)
我分了4個Part，每一個fold會取其中1/4當Valid set，提升蠻明顯的
我不確定在Fold外面多加一層迴圈能否增強效果，我的想法是每一個fold跑短一點防止overfit某一個fold，然後多跑幾個fold迴圈增加epoch

## Scheduler
老朋友`CosineAnnealingLR`，每跑8個fold會循環一次LR

