1. Directory Structure.
    The directory structure is as follows. (Images not inside the PA1 directory.)

    - PA1
    - tiny-imagenet-200

2. For experiment 1, we test all three models with batch size 32. You can run cells to run three models sequentially.

3. For experiment 2 and 3, we use ResNet18 with batch size 256. This leads huge performance degradation; but we are still able to compare
   the given regularization and optimization performances. 
   Again, you can run cells to run all three regularizations and optimizers sequentially. Note that we reset model for each technique.

4. Under each experiment cell, we also plot the training loss got from each iteration. This helps us see how the training went.