# HLS-LeNet
In the updating process...<br>
Hardware equipment ZYNQ 7035 SoC.<br>
Using AXI-master protocol to transmit weight with ARM.<br>
1.LeNet.cpp     —— The LeNet component code.<br>
2.LeNet.h       —— The header file.<br>
3.LeNet_tb.cpp  —— The testbench of the component.<br>

# Circuit(老图，连IP名称都写错了，LeNet写成LetNet...)
![Circuit](https://github.com/a2824256/HLS-LeNet/blob/master/circuit.png)

# LeNet-Weights-Exporter
Use GPU to train the network and derive weights.<br>
通过PC训练网络，然后将权重从多维转为一位数组通过ARM传输到FPGA的内部RAM
https://github.com/a2824256/LeNet-Weights-Exporter

# 相关文章
### XILINX SDK Xil_Out32传入float类型参数的解决方案
https://blog.csdn.net/a2824256/article/details/89409786

### XILINX HLS + Vivado + SDK实现通过AXI-Master协议从ARM(PS)传输数组到FPGA(PL)端RAM
https://blog.csdn.net/a2824256/article/details/89409671
