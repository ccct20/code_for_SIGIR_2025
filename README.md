# code_for_SIGIR2025
The BE4JM code

# BE4JM code for SIGIR-2025 anonymous review 
Title: Enhancing Dynamic Trust and Interest Prediction Through Balance Relationships Excitation

Paper ID: 516


In this paper, we reconsider the often overlooked informative relationships among unidentified social links and propose BE4JM to jointly model two types of user behavior on SNSs. Specifically, the proposed framework includes a comprehensive joint modeling framework, which includes a data augmentation module based on social balance theory, the recurrent neural networks for extracting dynamic embeddings, the time-aware multi-level neural attention networks to flexibly aggregate information from both node and graph perspectives, thereby enhancing the quality of embeddings, and an interactive module to mutually enhance the effectiveness of two prediction tasks. Extensive experiments demonstrate the effectiveness of our proposed framework.

<div align="center">
  <img src="https://github.com/ccct20/code_for_SIGIR_2025/blob/main/images/figure_4.jpg" alt="Description" width="280" height="200" />
  <img src="https://github.com/ccct20/code_for_SIGIR_2025/blob/main/images/figure3_new.jpg" alt="Description" width="450" height="200" />
</div>



# UPDATE 20250121:
We release or update the source code.


# Usage:

We implemented the code by Python2.7, users need to convert the code into a new one if the Python version 3.x.x is considered.
If you use python2.7.x, with tensorflow-gpu-1.12.0 (refer to the corresponding correct version of the graphics card) and tensorFlow-1.12.0 you can run the code in the directory of BE4JM. To avoid potential issues caused by device or version discrepancies, we kindly recommend running the CPU version to observe the results.

`entry.py --data_name=<data_name> --model_name=BE4JM --gpu=<gpu id>`

Following are the command examples with tensorflow-1.12.0:

`python entry.py --data_name=epinions --model_name=BE4JM`
