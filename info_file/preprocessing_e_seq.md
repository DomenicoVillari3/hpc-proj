# Preprocessing:
sul file ``` preprocess.py ``` è stato caricato il dataset ***Epsilon***.
-  è già normalizzato tra 0 ed 1 
-  le label hanno valore di [-1,1].
- Non viene effettuato alcuno split perchè già diviso in train e test nativamente. 
- le label sono castate a [0,1] 

``` y_train = ((y_train + 1) / 2).astype(np.float32)```
- sono salvati su file in formato ``` .bin ```
- le shape rilavate sono: 
    
    ``` Train: 400000 | Test: 100000 | D=2000```

- viene generato un file json con le shape 

    ``` {"N_train": 400000, "N_test": 100000, "D": 2000} ```

# Sequential SGD 



