import pickle

import matplotlib.pyplot as plt

acc_list_path = "../../resources/output/accuracy_list/acc_list_sample.p"
with open(acc_list_path, 'rb') as f:
    acc_list = pickle.load(f)

# acc_list_path = "../../resources/output/accuracy_list/acc_list_40000.p"
# with open(acc_list_path, 'rb') as f:
#     acc_list_40000 = pickle.load(f)
#
# acc_list_path = "../../resources/output/accuracy_list/acc_list_50000.p"
# with open(acc_list_path, 'rb') as f:
#     acc_list_50000 = pickle.load(f)
#
# acc_list_path = "../../resources/output/accuracy_list/acc_list_100000.p"
# with open(acc_list_path, 'rb') as f:
#     acc_list_100000 = pickle.load(f)
#
# plt.plot(acc_list_40000, label="40000 tables")
# plt.plot(acc_list_50000, label="50000 tables")
# plt.plot(acc_list_100000, label="100000 tables")


plt.plot(acc_list, label="sample")


plt.xlabel('No of iterations')
plt.ylabel('Accuracy in (%)')
plt.legend()
plt.savefig('../../resources/images/accuracy_sample.png', bbox_inches='tight')
plt.show()
