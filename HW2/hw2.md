# HW1: Classification

## Concat frames
一段音訊的資料很難在單個frame裡面表現，把多個frame接在一起能明顯改善。

## Network structure
多加幾層網路
另外，使用BN跟Dropout防止over fitting

## optimizer and scheduler
optimizer我用`AdamW`，似乎是更新更好的Adam
scheduler我用`CosineAnnealingLR`，也沒有為什麼，在查看 [Pytorch官方文檔](https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim) 時，看不同scheduler的description，挑了一個順眼看起來好用的。當然我也試過如`CosineAnnealingWarmRestarts`，但結果沒那麼好(但並不代表這個scheduler不好喔!)。

## Hyperparameters
+ `seed` Trial and Luck(???) 沒錯，`seed = 10270406`就是因為我在2023/10/27 4:06PM時測試，之後試了`10270422`之類的，結果分數差了很多:(
+ `Batch Size` 調低一點有幫助，上課要認真聽喔
+ `Learning rate` depends on the optimizer. 在嘗試後，發現5e-5最好。
+ `early_stop` 不能設太多，不然會Overfitting，因為其實在valid set裡面找loss低的也是一種train，多幾個step找出來的參數可能只適合valid set，不一定代表說預測越準。
+ `valid_ratio` 太大太小都不行，發現0.2最好

## 參考資料:
> https://hackmd.io/@swlearning/Bk_U4jev5
> https://github.com/Joshuaoneheart/ML2022_all_A_plus