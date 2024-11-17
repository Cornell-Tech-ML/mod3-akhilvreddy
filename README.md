# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

------

## Training With CPU

First, I trained the 'split' dataset with the following hyperparameters: 

```
python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05
```

This resulted in the following logs: 

```
Epoch  10  loss  4.796329464539841 correct 41
Epoch  20  loss  4.152904790956639 correct 41
Epoch  30  loss  3.3861868525728545 correct 42
Epoch  40  loss  2.995197988227874 correct 47
Epoch  50  loss  2.0617537019290606 correct 48
Epoch  60  loss  2.5601062571032442 correct 47
Epoch  70  loss  1.6356268174226503 correct 48
Epoch  80  loss  1.4878333520197478 correct 50
Epoch  90  loss  0.8664745156084035 correct 50
Epoch  100  loss  0.5197747905772951 correct 49
Epoch  110  loss  1.3106387971315339 correct 50
Epoch  120  loss  1.771440004125864 correct 50
Epoch  130  loss  1.690837001725669 correct 50
Epoch  140  loss  0.7571167510195219 correct 50
Epoch  150  loss  1.421010793341866 correct 49
Epoch  160  loss  1.013126772317953 correct 49
Epoch  170  loss  1.2172516925947825 correct 50
Epoch  180  loss  0.6646588655456165 correct 49
Epoch  190  loss  0.7372400843404222 correct 49
Epoch  200  loss  0.47002058033373806 correct 49
Epoch  210  loss  0.30938110676252584 correct 50
Epoch  220  loss  1.6593963485921421 correct 50
Epoch  230  loss  0.5341722241615815 correct 50
Epoch  240  loss  0.6450703712247884 correct 50
Epoch  250  loss  0.19983239230180788 correct 49
Epoch  260  loss  0.17725232201117022 correct 49
Epoch  270  loss  0.6707024203930815 correct 49
Epoch  280  loss  0.08585751411749706 correct 50
Epoch  290  loss  0.35937025397361105 correct 50
Epoch  300  loss  0.2557235919643731 correct 50
Epoch  310  loss  0.2773810068723875 correct 49
Epoch  320  loss  1.3193540493282303 correct 50
Epoch  330  loss  0.03054563841028581 correct 49
Epoch  340  loss  0.2160021232553947 correct 49
Epoch  350  loss  0.029580950955366263 correct 50
Epoch  360  loss  0.2387157379774431 correct 49
Epoch  370  loss  0.33253677047760494 correct 50
Epoch  380  loss  0.10270748609073295 correct 50
Epoch  390  loss  0.15319086453761108 correct 50
Epoch  400  loss  0.09949738754490794 correct 50
Epoch  410  loss  0.15899610539900433 correct 50
Epoch  420  loss  0.6351888779126083 correct 49
Epoch  430  loss  0.7415385902335059 correct 49
Epoch  440  loss  0.17956285233642384 correct 49
Epoch  450  loss  0.1501994692349236 correct 49
Epoch  460  loss  0.15702987004405768 correct 49
Epoch  470  loss  0.10261295152412003 correct 50
Epoch  480  loss  0.06817588209354158 correct 50
Epoch  490  loss  0.05214814642605661 correct 50
Epoch  500  loss  0.03721497088592909 correct 50
```

We can see that it classified 50/50 correctly, with a training time of 104.98 seconds and an average time per epoch is 0.21 seconds. 

---

The next dataset I trained was 'xor' with the following hyperparameters:

```
python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05
```

This resulted in the following logs: 

```
Epoch  10  loss  5.556472348667344 correct 45
Epoch  20  loss  4.800427718047741 correct 41
Epoch  30  loss  3.74663797856912 correct 41
Epoch  40  loss  4.779327767167111 correct 45
Epoch  50  loss  3.6647247582989135 correct 45
Epoch  60  loss  4.6333102961906905 correct 44
Epoch  70  loss  4.982616331218322 correct 45
Epoch  80  loss  2.0032474752899065 correct 45
Epoch  90  loss  3.917489618072344 correct 45
Epoch  100  loss  3.680510190598146 correct 46
Epoch  110  loss  0.8584517394550654 correct 46
Epoch  120  loss  1.671421950261504 correct 46
Epoch  130  loss  1.9676920030071425 correct 46
Epoch  140  loss  3.2837915476739514 correct 47
Epoch  150  loss  5.113663538989956 correct 46
Epoch  160  loss  2.7631067483996485 correct 47
Epoch  170  loss  4.278842093517767 correct 44
Epoch  180  loss  3.408831722480358 correct 45
Epoch  190  loss  1.6929444678042933 correct 47
Epoch  200  loss  2.401486547174001 correct 48
Epoch  210  loss  1.0747027338312272 correct 48
Epoch  220  loss  1.3107558957050085 correct 46
Epoch  230  loss  0.853949151027406 correct 48
Epoch  240  loss  1.0089817722269394 correct 48
Epoch  250  loss  3.799455044061256 correct 46
Epoch  260  loss  1.0674571591292668 correct 49
Epoch  270  loss  1.1201256548180132 correct 48
Epoch  280  loss  2.5956980146056283 correct 49
Epoch  290  loss  0.5207738124114579 correct 49
Epoch  300  loss  1.0521451892397315 correct 49
Epoch  310  loss  1.9122280007401469 correct 50
Epoch  320  loss  1.4567531754204992 correct 49
Epoch  330  loss  1.1009087960728507 correct 49
Epoch  340  loss  0.632658593433779 correct 50
Epoch  350  loss  1.6325997494081985 correct 50
Epoch  360  loss  1.3008115489934704 correct 49
Epoch  370  loss  0.4533859646720788 correct 50
Epoch  380  loss  0.589683177733833 correct 48
Epoch  390  loss  1.3522639549348305 correct 49
Epoch  400  loss  1.6168599095545837 correct 49
Epoch  410  loss  1.5907328823121943 correct 49
Epoch  420  loss  1.5648937425613297 correct 49
Epoch  430  loss  1.5393424903009898 correct 49
Epoch  440  loss  1.5140791255311746 correct 49
Epoch  450  loss  1.4891036482518837 correct 49
Epoch  460  loss  1.4644160584631172 correct 49
Epoch  470  loss  1.440016356164875 correct 49
Epoch  480  loss  1.4159045413571573 correct 49
Epoch  490  loss  1.3920806140399643 correct 49
Epoch  500  loss  1.369289573812158 correct 50
```

This classified 50/50 correctly, with a training time of 103.92 seconds and an average time per epoch of 0.20784 seconds. 

## Training With GPU
Similar to the CPU case, I trained my GPU based models with the split and xor data set with the same hyperparameters.

For the split case, I used the following command with these hyperparameters:

```
python run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05
```

which gave me the following logs: 

```
Epoch  10  loss  7.019764740021523 correct 26
Epoch  20  loss  6.004594951977885 correct 26
Epoch  30  loss  5.823311493868308 correct 29
Epoch  40  loss  6.104026871469294 correct 29
Epoch  50  loss  6.76482314228329 correct 29
Epoch  60  loss  5.911336066655572 correct 33
Epoch  70  loss  5.705281972944201 correct 34
Epoch  80  loss  5.595681561188602 correct 31
Epoch  90  loss  5.1745161863112346 correct 34
Epoch  100  loss  5.497527633258522 correct 37
Epoch  110  loss  4.731929121692181 correct 37
Epoch  120  loss  5.441762711179047 correct 34
Epoch  130  loss  2.9445313321295457 correct 37
Epoch  140  loss  2.5272146631655397 correct 37
Epoch  150  loss  2.073909243001914 correct 38
Epoch  160  loss  2.2705347787176056 correct 37
Epoch  170  loss  2.7599187173776016 correct 37
Epoch  180  loss  2.0661751232040277 correct 38
Epoch  190  loss  1.3121062895333089 correct 37
Epoch  200  loss  1.6729661549290453 correct 38
Epoch  210  loss  0.7744932553047947 correct 37
Epoch  220  loss  0.5500062036802862 correct 38
Epoch  230  loss  2.29404707692132 correct 40
Epoch  240  loss  0.5002091966811272 correct 40
Epoch  250  loss  0.9425609913378616 correct 40
Epoch  260  loss  2.2304019190790565 correct 41
Epoch  270  loss  0.31760381310246144 correct 40
Epoch  280  loss  1.0310527013220272 correct 40
Epoch  290  loss  1.9288578021775815 correct 40
Epoch  300  loss  2.932291688995339 correct 41
Epoch  310  loss  1.1825689279154183 correct 40
Epoch  320  loss  1.4439838568546945 correct 41
Epoch  330  loss  1.4360821992015578 correct 40
Epoch  340  loss  1.8379901198751734 correct 40
Epoch  350  loss  0.6007009063034967 correct 41
Epoch  360  loss  0.6347969503506936 correct 40
Epoch  370  loss  0.20294623280189952 correct 41
Epoch  380  loss  0.18839950753651663 correct 41
Epoch  390  loss  0.8263870082021266 correct 40
Epoch  400  loss  1.9609313160273658 correct 41
Epoch  410  loss  1.3725595891437348 correct 40
Epoch  420  loss  1.7520451058135569 correct 41
Epoch  430  loss  1.3187472543497418 correct 40
Epoch  440  loss  1.4371711033718118 correct 42
Epoch  450  loss  1.2462464888755092 correct 43
Epoch  460  loss  1.6140853564862032 correct 42
Epoch  470  loss  0.2660051480290557 correct 45
Epoch  480  loss  2.6149538775307235 correct 41
Epoch  490  loss  0.05471673228268411 correct 46
Epoch  500  loss  0.0268411547130735 correct 47
```

This classified 47/50 correctly, with a training time of 73.29 seconds and an average time per epoch of 0.14658 seconds. 

As we can see, training with a GPU is much faster than training with a CPU. The drop in accuracy could be correlated with slight errors in 'matmul'. 

--- 
I then moved on to the xor case with the same hyperparameters. 

```
python run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05
```

which gave me the following logs: 

```
Epoch 10  loss 6.803629338296769 correct 15
Epoch 20  loss 5.978945042726535 correct 17
Epoch 30  loss 6.099586241992977 correct 27
Epoch 40  loss 5.209064630823643 correct 30
Epoch 50  loss 6.311235938855277 correct 29
Epoch 60  loss 5.743096755824945 correct 30
Epoch 70  loss 4.038710018031654 correct 35
Epoch 80  loss 4.174407380442765 correct 38
Epoch 90  loss 3.993671572658741 correct 32
Epoch 100  loss 2.6921446903770403 correct 33
Epoch 110  loss 3.545951526324117 correct 35
Epoch 120  loss 2.142818620175353 correct 32
Epoch 130  loss 2.651703781161006 correct 32
Epoch 140  loss 2.32651229379064 correct 38
Epoch 150  loss 2.2038480853988127 correct 37
Epoch 160  loss 0.7066599431418727 correct 36
Epoch 170  loss 3.3542934516420027 correct 32
Epoch 180  loss 2.4826432747843707 correct 40
Epoch 190  loss 0.9481890575776962 correct 37
Epoch 200  loss 2.9156248094027677 correct 36
Epoch 210  loss 0.7625276995263996 correct 39
Epoch 220  loss 0.7201629489304313 correct 38
Epoch 230  loss 1.5608168331745642 correct 37
Epoch 240  loss 1.042123323191709 correct 38
Epoch 250  loss 0.6637275699753559 correct 38
Epoch 260  loss 1.8398900144000954 correct 39
Epoch 270  loss 1.8263562339863402 correct 41
Epoch 280  loss 0.7477798371592537 correct 44
Epoch 290  loss 2.1515050901038903 correct 39
Epoch 300  loss 1.4250216387066 correct 43
Epoch 310  loss 1.657507729836587 correct 42
Epoch 320  loss 0.7718371934703561 correct 44
Epoch 330  loss 1.012462068437372 correct 44
Epoch 340  loss 0.8012189319243602 correct 39
Epoch 350  loss 1.1644494862281338 correct 43
Epoch 360  loss 2.147014817199452 correct 39
Epoch 370  loss 0.4679137691924435 correct 40
Epoch 380  loss 0.754790564120551 correct 39
Epoch 390  loss 0.46057855817986815 correct 43
Epoch 400  loss 1.0442401320782517 correct 42
Epoch 410  loss 0.8448420090561486 correct 43
Epoch 420  loss 0.5866124606279399 correct 43
Epoch 430  loss 0.3694246421785613 correct 44
Epoch 440  loss 0.6765883468205739 correct 45
Epoch 450  loss 0.20510044746534808 correct 45
Epoch 460  loss 1.0763548219712589 correct 43
Epoch 470  loss 1.7462068321252602 correct 43
Epoch 480  loss 0.4084942652473577 correct 46
Epoch 490  loss 0.586447477336259 correct 47
Epoch 500  loss 0.586447477336259 correct 47
```

This classified 47/50 correctly, with a training time of 80.46 seconds and an average time per epoch of 0.16092 seconds. 