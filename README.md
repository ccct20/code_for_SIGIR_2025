# code_for_AAAI2025
The BRAJM code

# BRAJM code for AAAI-2025 anonymous review 
Title: Enhancing Dynamic Trust and Interest Prediction Through Balance Relationships Augmentation

Paper ID: 10207

![](https://github.com/ccct20/code_for_AAAI2025/blob/main/images/figure3_new.jpg)

In this paper, we reconsider the often overlooked informative relationships among unidentified social links and propose BRAJM to jointly model two types of user behavior on SNSs. Specifically, the proposed framework includes a comprehensive joint modeling framework, which includes a data augmentation module based on social balance theory, the recurrent neural networks for extracting dynamic embeddings, the time-aware multi-level neural attention networks to flexibly aggregate information from both node and graph perspectives, thereby enhancing the quality of embeddings, and an interactive module to mutually enhance the effectiveness of two prediction tasks. Extensive experiments demonstrate the effectiveness of our proposed framework.

![](https://github.com/ccct20/code_for_AAAI2025/blob/main/images/figure2_new.jpg)

# UPDATE 20240815:
We release or update the source code.


# Usage:

We implemented the code by Python2.7, users need to convert the code into a new one if the Python version 3.x.x is considered.
If you use python2.7.x, with tensorflow-gpu-1.12.0 (refer to the corresponding correct version of the graphics card) and tensorFlow-1.12.0 you can run the code in the directory of BRAJM. To avoid potential issues caused by device or version discrepancies, we kindly recommend running the CPU version to observe the results.

`entry.py --data_name=<data_name> --model_name=BRAJM --gpu=<gpu id>`

Following are the command examples with tensorflow-1.12.0:

`python entry.py --data_name=Epinions --model_name=BRAJM`
