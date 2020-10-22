import numpy as np
import sys
import pickle
import os
sys.path.append(os.path.abspath('..'))
from symbolic_pursuit.models import SymbolicRegressor
from datasets.data_loader_UCI import data_loader, mixup
from sklearn.metrics import mean_squared_error
from experiments.train_model import train_model


black_box = sys.argv[1]
data_name = sys.argv[2]
n_model = 5
data_list = []  # List containing the data used in each experiment
random_list = []
print("Welcome to this experiment evaluating the performance of faithful modeling. \n"
      + "This experiment uses the black-box " + black_box + " on the dataset " + data_name + ".")
print(100*"%" + "\n" + 100*"%" + "\n" + "Now building the black-box models.\n" + 100*"%" + "\n" + 100*"%")
for n in range(n_model):
    X_train, y_train, X_test, y_test = data_loader(data_name)
    data_list.append([X_train, y_train, X_test, y_test])
    X_random = mixup(X_train)
    random_list.append(X_random)

model_list = []  # List containing each model
for n in range(n_model):
    print("Now working on model", n + 1, "/", n_model)
    X_train, y_train, _, _ = data_list[n]
    model = train_model(X_train, y_train, black_box=black_box)
    model_list.append(model)

faithful_list = []  # List containing each faithful model
print(100*"%" + "\n" + 100*"%" + "\n" + "Now building the faithful models.\n" + 100*"%" + "\n" + 100*"%")
for n in range(n_model):
    print("Now working on model", n+1, "/", n_model)
    X_random = random_list[n]
    model = model_list[n]
    faithful_model = FaithfulModel(verbosity=False)
    faithful_model.fit(model.predict, X_random)
    faithful_list.append(faithful_model)

print(100*"%" + "\n" + 100*"%" + "\n" + "Now computing the statistics.\n" + 100*"%" + "\n" + 100*"%")
model_errors = []  # Generalization MSE of the Black-Box
faithful_errors = []  # Generalization MSE of the Faithful-Model
distance_faithfulBB = []  # Generalization Distance between the Faithful-Model and the BB
faithful_nterms = []  # Number of terms of the Faithful Model


for n in range(n_model):
    _, _, X_test, y_test = data_list[n]
    model_errors.append(mean_squared_error(y_test, model_list[n].predict(X_test)))
    faithful_errors.append(mean_squared_error(y_test, faithful_list[n].predict(X_test)))
    faithful_nterms.append(len(faithful_list[n].terms_list))
    distance_faithfulBB.append(mean_squared_error(model_list[n].predict(X_test),
                                                  faithful_list[n].predict(X_test)))

model_avgMSE, model_stdMSE = np.average(model_errors), np.std(model_errors)
faithful_avgMSE, faithful_stdMSE = np.average(faithful_errors), np.std(faithful_errors)
faithful_avgNterms, faithful_stdNterms = np.average(faithful_nterms), np.std(faithful_nterms)
faithful_avgDist, faithful_stdDist = np.average(distance_faithfulBB), np.std(distance_faithfulBB)

# Print and save the results

output_file = open("experiments/" + black_box + "_" + data_name + ".txt", "w")
output_file.write(100*"=" + "\n")
print("Black-Box generalization MSE", model_avgMSE, "+/-", model_stdMSE)
output_file.write("Black-Box generalization MSE: " + str(model_avgMSE) + " +/- " + str(model_stdMSE) + "\n")
print("Faithful generalization MSE", faithful_avgMSE, "+/-", faithful_stdMSE)
output_file.write("Faithful generalization MSE: " + str(faithful_avgMSE)
                  + " +/- " + str(faithful_stdMSE) + "\n")
print("Generalization distance between the Faithful model and the Black-Box: ", faithful_avgDist,
      "+/-", faithful_stdDist)
output_file.write("Generalization distance between the Faithful model and the Black-Box: "
                  + str(faithful_avgDist) + " +/- " + str(faithful_stdDist) + "\n")
print("Training Faithful Model number of terms", faithful_avgNterms, "+/-", faithful_stdNterms)
output_file.write("Training Faithful Model number of terms: " + str(faithful_avgNterms) + " +/- "
                  + str(faithful_stdNterms) + "\n")

faithful_bestID = int(np.argmin(faithful_errors))
faithful_worstID = int(np.argmax(faithful_errors))
print(100*'%')
print("Best Faithful Model:", faithful_list[faithful_bestID].get_expression())
faithful_list[faithful_bestID].print_projections()
output_file.write("Best Faithful Model: " + str(faithful_list[faithful_bestID]) + "\n"
                  + str(faithful_list[faithful_bestID].string_projections()))
print("Associated generalization loss: ", faithful_errors[faithful_bestID])
output_file.write("Associated loss: " + str(faithful_errors[faithful_bestID]) + "\n")
print("Worst Faithful Model:", faithful_list[faithful_worstID].get_expression())
faithful_list[faithful_worstID].print_projections()
output_file.write("Worst Faithful Model: " + str(faithful_list[faithful_worstID]) + "\n"
                  + str(faithful_list[faithful_worstID].string_projections()))
print("Associated generalization loss: ", faithful_errors[faithful_worstID])
output_file.write("Associated loss: " + str(faithful_errors[faithful_worstID]) + "\n")
print(100*'%')
output_file.close()

# Save everything

with open("experiments/" + black_box + "_" + data_name + ".pickle", 'wb') as filename:
    save_tuple = (model_list, faithful_list, model_errors, faithful_errors, faithful_nterms,
                  data_list, random_list)
    pickle.dump(save_tuple, filename)

