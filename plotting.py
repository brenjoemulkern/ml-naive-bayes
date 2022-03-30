from cv2 import log
import pandas as pd
import matplotlib.pyplot as plt

# Naive Bayes Beta Values Acc Plot

# nb_data = pd.read_csv('nb_val_results/nb_acc.csv')

# plt.plot(nb_data['beta'], nb_data['acc'])
# plt.xscale('log')
# plt.title('Naive Bayes Accuracy')
# plt.xlabel('Beta Value')
# plt.ylabel('Accuracy')
# plt.savefig('nb_val_results/nb_acc_plot.png')

lr_eta_data = pd.read_csv('lr_val_results/lr_acc_eta.csv')

plt.plot(lr_eta_data['eta'], lr_eta_data['acc'])
plt.title('Logistic Regression Accuracy: Lambda = 0.01')
plt.xlabel('Eta Value')
plt.ylabel('Accuracy')
plt.savefig('lr_val_results/lr_acc_eta_plot.png')

# lr_lamb_data = pd.read_csv('lr_val_results/lr_acc_lamb.csv')

# plt.plot(lr_lamb_data['lambda'], lr_lamb_data['acc'])
# plt.title('Logistic Regression Accuracy: Eta = 0.01')
# plt.xlabel('Lambda Value')
# plt.ylabel('Accuracy')
# plt.savefig('lr_val_results/lr_acc_lamb_plot.png')