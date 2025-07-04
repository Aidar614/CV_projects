# 🕵️ Shoplifting Detection Using a Video Classifier

This project is a system for detecting shoplifting using computer vision.

It is based on the VideoMAE neural network model, trained for video classification. The model analyzes short 16-frame video fragments and determines whether theft or normal behavior occurs in them.

The system integrates with the YOLO object detector, which finds and tracks people in the frame, after which their actions are transferred to the classifier for analysis.

[Jupyter Notebook](./Shoplifting_Detection.ipynb)

## ✅ Results: Shoplifting Detected
<table>
  <tr>
    <td><img src="./test_1_1.gif" width="500"></td>
    <td><img src="./test_6_6.gif" width="500"></td>
  </tr>
  <tr>
    <td><img src="./test_7_7.gif" width="500"></td>
    <td><img src="./test_4_4.gif" width="500"></td>
  </tr>
</table>

## 🤔 Challenging Case: Placing Items in a Basket
<table>
  <tr>
    <td><img src="./test_2_2.gif" width="500"></td>
    <td><img src="./test_3_3.gif" width="500"></td>
  </tr>
</table>