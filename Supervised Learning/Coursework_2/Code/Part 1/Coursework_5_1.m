clear all;
%import data file
Data = dlmread('zipcombo.dat');

%define parameters
epoch = 3;

%obtain the size of the dataset
[Row,Column] = size(Data);

%used to store the predicted value of Y
Yvalue = zeros(7438,1);
%Yt_predict:predict whether the value is larger than 0
Yt_predict = zeros(7438,10);
signTrain = zeros(7438,10);
Yt_predict_test = zeros(1860,10);

%for calculate the expected Y
Y_09=meshgrid(0:9,1:7438);

meanTrain = zeros(7,20);
meanTest= zeros(7,20);

number = 0:9;
row = (ones(10,1)*number)';
column = ones(10,1)*number;
matrix = abs(row-column);   

%% training
for run = 1:20
    A=randperm(Row);
    %use the fisrt 7438 rows in A as the trainning set
    Xtrain = Data (A(1:7438),2:257);
    Ytrain = Data (A(1:7438),1);
    %use the next 1860 rows in A as the testing set
    Xtest = Data (A(7439:9298),2:257);
    Ytest = Data (A(7439:9298),1);

    %find the expected value
    Y(:,:) = Y_09(:,:) - Ytrain(:).*ones(1,10);
    Y(Y~=0)=-1;
    Y(Y==0)=1;
    
    for c=1:9
        x1 = repmat(sum(Xtrain.^2,2),[1,7438]);
        y1 = repmat(sum(Xtrain.^2,2)',[7438,1]);
        K = exp(-(x1+y1-2*(Xtrain*Xtrain'))/(2*c^2));
        alpha = zeros(7438,10);

        for Epoch = 1:5
            for t = 1:7438
                %calculate the predict Y in training dataset
                Yt_predict(t,:) = transpose(alpha(:,:))*K(:,t);
                signTrain(t,:) = sign(Yt_predict(t,:)-0.000000001);

                %update the alpha value
                for n = 1:10
                    %update the alpha value if values are not the same
                    if Y(t,n) * Yt_predict(t,n) <=0
                         alpha(t,n) = alpha(t,n) - signTrain(t,n);
                    end
                end

               %predict the value of y
               [m,n] = max(Yt_predict(t,:));
               Yvalue(t) = n-1;
            end
                mistakeTrainMatrix = Yvalue - Ytrain;
                mistakeTrain = nnz(mistakeTrainMatrix);
        end
        
     %% testing
        % Calculate the predict value in test dataset
        x2 = repmat(sum(Xtrain.^2,2),[1,1860]);
        y2 = repmat(sum(Xtest.^2,2)',[7438,1]);
        K2 = exp(-(x2+y2-2*Xtrain*Xtest')/(2*c^2));
        Yt_predict_test = transpose(transpose(alpha(:,:))*K2);

        %predict the value of Y_test
        [m_test,Yvalue_test] = max(Yt_predict_test,[],2);
        mistakeTestMatrix = Yvalue_test - Ytest - 1;
        mistakeTest = nnz(mistakeTestMatrix);

        meanTrain(c,run) = mistakeTrain/7438;
        meanTest(c,run)= mistakeTest/1860;
    end
end

result_Train = zeros(c,2);
result_Train(:,1)=mean(meanTrain,2);
result_Train(:,2)=std(meanTrain,0,2);

result_Test = zeros(c,2);
result_Test(:,1)=mean(meanTest,2);
result_Test(:,2)=std(meanTest,0,2);

