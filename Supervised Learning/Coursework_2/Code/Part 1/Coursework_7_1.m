clear all;
%import data file
Data = dlmread('zipcombo.dat');
mistakeTrain = zeros(7,20);
mistakeTest = zeros(7,20);
[Row,Column] = size(Data);
for run = 1:20
    A=randperm(Row);
    %use the fisrt 7438 rows in A as the trainning set
    Xtrain = Data (A(1:7438),2:257);
    Ytrain = Data (A(1:7438),1);
    %use the next 1860 rows in A as the testing set
    Xtest = Data (A(7439:9298),2:257);
    Ytest = Data (A(7439:9298),1);
    
    for sigma = 5:10
        
        %call the statement templateSVM to define kernel
        t = templateSVM('KernelFunction','gaussian','KernelScale',sigma);
        %call the statement fitcecoc to calculate classifier
        model = fitcecoc(Xtrain,Ytrain,'Learners',t);
        TrainPredictValue = predict(model,Xtrain);
        TestPredictValue = predict(model,Xtest);
        %calculate mistakes
        mistakeTrain(sigma-4,run) = nnz(Ytrain - TrainPredictValue)/7438;
        mistakeTest(sigma-4,run) = nnz(Ytest - TestPredictValue)/1860;
    end 
end    

%Store mean and std of Train
result_Train = zeros(7,2);
result_Train(:,1)=mean(mistakeTrain,2);
result_Train(:,2)=std(mistakeTrain,0,2);

%Store mean and std of Test
result_Test = zeros(7,2);
result_Test(:,1)=mean(mistakeTest,2);
result_Test(:,2)=std(mistakeTest,0,2);