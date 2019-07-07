clear all;
%import data file
Data = dlmread('zipcombo.dat');

%Define epoch
epoch = 10;

%Obtain the size of the dataset
[Row,Column] = size(Data);

%Used to store the predicted value of Y
Yvalue = zeros(7438,1);
%Yt_predict:predict whether the value is larger than 0
Yt_predict = zeros(7438,10);
signTrain = zeros(7438,10);
Yt_predict_test = zeros(1860,10);

%for calculate the expected Y
Y_09=meshgrid(0:9,1:7438);

%store mean values in d=1:7 and 20 runs
meanTrain = zeros(7,20);
meanTest= zeros(7,20);

%% training
for run = 1:20
    A=randperm(Row);
    %use the fisrt 7438 rows in A as the trainning set
    Xtrain = Data (A(1:7438),2:257);
    Ytrain = Data (A(1:7438),1);
    %use the next 1860 rows in A as the testing set
    Xtest = Data (A(7439:9298),2:257);
    Ytest = Data (A(7439:9298),1);
 
    %Find the expected value
    Y(:,:) = Y_09(:,:) - Ytrain(:).*ones(1,10);
    Y(Y~=0)=-1;
    Y(Y==0)=1;
    
    for d=1:7
        %calculate kernel in each d
        K = (Xtrain(:,:)*Xtrain(:,:)').^d;
        transK = K';
        alpha = zeros(7438,10);

        %calculate alpha row by row
        for Epoch = 1:5
            for t = 1:7438
                %calculate the predict Y in training dataset
                Yt_predict(t,:) = transpose(alpha(:,:))*transK(:,t);                  
                signTrain(t,:) = sign(Yt_predict(t,:)-0.000000001);

                %update the alpha value if values are not the same
                for n = 1:10
                    if Y(t,n) * Yt_predict(t,n) <=0
                         alpha(t,n) = alpha(t,n) - signTrain(t,n);
                    end
                end

               %Find the expected value of Y
               [m,n] = max(Yt_predict(t,:));
               Yvalue(t) = n-1;
            end
            %Calculate raw totals of mistake train
            mistakeTrainMatrix = Yvalue - Ytrain;
            mistakeTrain = nnz(mistakeTrainMatrix);
        end       
 %% testing
        % Calculate the predict value in test dataset
        Yt_predict_test(:,:) = transpose(transpose(alpha(:,:))*transpose((Xtest(:,:)* transpose(Xtrain(:,:))).^d));

        %Calculate raw totals of mistake test
        [m_test,Yvalue_test] = max(Yt_predict_test,[],2);
        mistakeTestMatrix = Yvalue_test - Ytest - 1;
        mistakeTest = nnz(mistakeTestMatrix);
        
        %Change the mistake in train an test to percentages
        meanTrain(d,run) = mistakeTrain/7438;
        meanTest(d,run)= mistakeTest/1860;
    end
end

%Store mean and std of Train
result_Train = zeros(7,2);
result_Train(:,1)=mean(meanTrain,2);
result_Train(:,2)=std(meanTrain,0,2);

%Store mean and std of Test
result_Test = zeros(7,2);
result_Test(:,1)=mean(meanTest,2);
result_Test(:,2)=std(meanTest,0,2);

