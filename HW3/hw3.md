  File "d:\_Code\ML2022-Spring-HW\HW3\2022ml_hw3_image_classification.py", line 182, in <module>
    loss.backward()
  File "C:\Users\xTaiwan\AppData\Local\Programs\Python\Python39\lib\site-packages\torch\_tensor.py", line 487, in backward
    torch.autograd.backward(
  File "C:\Users\xTaiwan\AppData\Local\Programs\Python\Python39\lib\site-packages\torch\autograd\__init__.py", line 197, in backward
    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling `cublasSgemm( handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc)`

**device = cpu**


Target -1 is out of bounds.
  File "D:\_Code\ML2022-Spring-HW\HW3\hw3.py", line 188, in <module>
    loss = criterion(logits, labels)

debugger labels: tensor([-1])
check dataloader and dataset

int(fname.split("/")[-1].split("\\")[-1].split("_")[0])
the lable parser is incorrect

Basicly VGG11
https://debuggercafe.com/wp-content/uploads/2021/04/vgg.jpg
BN Dropout added

Transform: 