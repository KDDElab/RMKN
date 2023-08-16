function[a,b] =test_model(test_data,model,kernels,JNN,KNN,number)
test.X = test_data;
test_data = cell(1,4);
 test_data{1} = test;
 test_data{2} = test;
 test_data{3} = test;
 test_data{4} = test;
parameters = graphJKNN_parameter();
parameters.ker = kernels;
parameters.nor.dat = {'true', 'true','true'};
parameters.nor.ker = {'true', 'true','true'};

output = JKNN_test(test_data, model,JNN,KNN,number);
a = output.predictedlabel;
b = output.outlier_score;
end