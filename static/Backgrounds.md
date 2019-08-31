**Backgrounds**



在Full-Text场景中，用户没有信息泄漏，但是Full-Text的效率太低。相比之下Part-Text效率较高，但是会泄漏用户评价了哪些item。

在真实场景中，用户评价了哪些item是非常重要的数据，使用这部分数据可以进行以下的数据推断：

1. Cyber criminals can leverage user attributes to perform targeted social engi- neering attacks (now often referred to as spear phishing attacks) and attacking personal- information-based backup authentication;
2. online social network providers and advertisers could use the user attributes for targeted advertisements;
3. Data brokers make profit via selling the user attribute information to other parties such as advertisers, banking companies, and insurance industries [18];
4. Surveillance agency can use the attributes to identify users and monitor their activities.



**Problem Definition**

| Notation | Description                                          |
| -------- | ---------------------------------------------------- |
| X        | user rated item set                                  |
| X'       | obfuscated user rated item set                       |
| y        | ground truth of users' attribute                     |
| y'       | prediction of user's attribute                       |
| D        | function to measure the increased time from X to X'  |
| A        | function to measure the attribute's prediction error |
| F        | function used for predicting y' from X               |
| O        | function used for obfuscating X                      |

In the part-test setting, denote X as the user rated item-set that exposed to the server. Assuming the server has a method, denoted as F, to inference users' attributes like gender, location, major or occupation. The inference process can be represented as y'=F(X), where y' is the prediction of users' attributes. The accuracy of y' is denoted as A(y, y'). Now we want to develop an obfuscation method O, such that X’=O(X), A(F(X’), y) is minimized and D(X, X’) is minimized, which will be:







问题描述

1. Full-Text可以保证用户的所有信息，但是效率太低；所以可以考虑针对part-text进行改进
2. 在Part-Text中，主要泄漏的用户隐私是 哪些Item被评价了，但是这个信息也是很重要的，因为可以用来做attribute inference attack
3. 所以问题可以定义为，在Part-Text的场景下，如何通过上传冗余的item信息，使得属性推导的难度增加、准确率下降，同时保证上传的冗余信息最少，以保证通信效率



待解决问题：

1. 属性推断是如何进行的，有没有数据集可以使用
2. 如果用户上传了-V，该怎么处理？
3. 是否可以让每个user-server之间都有一个秘钥对？