clear all;
%import data file
Data = importdata('zipcombo.dat');
mistakeTrain = zeros(20,7);
mistakeTest = zeros(20,7);
[Row,Column] = size(Data);

for run = 1:20
    A=randperm(Row);
    %use the fisrt 7438 rows in A as the trainning set
    Xtrain = Data (A(1:7438),2:257);
    Ytrain = Data (A(1:7438),1);
    %use the next 1860 rows in A as the testing set
    Xtest = Data (A(7439:9298),2:257);
    Ytest = Data (A(7439:9298),1);
    
    %Call Random Forests in libraries
    model = fitensemble(Xtrain,Ytrain,'bag',200,'tree','type','Classification');
    
    TrainPredictValue = predict(model,Xtrain);
    TestPredictValue = predict(model,Xtest);
    %calculate mistakes
    mistakeTrain(run) = nnz(Ytrain-TrainPredictValue);
    mistakeTest(run) = nnz(Ytest - TestPredictValue);
end 

%Store mean and std of Train
result_Train = zeros(7,2);
result_Train(:,1)=mean(mistakeTrain,2);
result_Train(:,2)=std(mistakeTrain,0,2);

%Store mean and std of Test
result_Test = zeros(7,2);
result_Test(:,1)=mean(mistakeTest,2);
result_Test(:,2)=std(mistakeTest,0,2);