clear all;
%import data file
Data = dlmread('zipcombo.dat');

%obtain the size of the dataset
[Row,Column] = size(Data);

%used to store the predicted value of Y
Yvalue = zeros(7440,1);
%Yt_predict:predict whether the value is larger than 0
Yt_predict = zeros(7440,10);
signTrain = zeros(7440,10);
Yt_predict_test = zeros(1858,10);

%for calculate the expected Y
Y_09=meshgrid(0:9,1:7440);

%store mean values in d=1:7 and 20 runs
dimension = zeros(1,20);
meanTest = zeros(1,20);

%% training
for run = 1:20
    A=randperm(Row);
    %use the fisrt 7440 rows in A as the trainning set
    Xtrain = Data (A(1:7440),2:257);
    Ytrain = Data (A(1:7440),1);
    %use the next 1858 rows in A as the testing set
    Xtest = Data (A(7441:9298),2:257);
    Ytest = Data (A(7441:9298),1);

    %find the expected value
    Y(:,:) = Y_09(:,:) - Ytrain(:).*ones(1,10);
    Y(Y~=0)=-1;
    Y(Y==0)=1;

    %build 5 cell for 5-fold cross-validation
    X_fold = cell(5,1);Y_fold = cell(5,1);Y_fold_real = cell(5,1);
    for i = 1 : 5
        X_fold{i} = Xtrain(1448* (i-1) + 1 : 1448 * i,:);
        Y_fold{i} = Y(1448* (i-1) + 1 : 1448 * i,:);
        Y_fold_real{i} = Ytrain(1448* (i-1) + 1 : 1448 * i);
    end

    mean_fold = zeros(7,1);
    for d = 1:7
        %assign values to alpha
        alpha_fold = cell(4,1);K_fold = cell(4,1);
        Yvalue_fold = zeros(1448,1);
        for i = 1:4
            alpha_fold{i} = zeros(1448,10);
            K_fold{i} = (X_fold{i}*X_fold{i}').^d;
        end

        Yt_fold_predict = zeros(1448,10);
        signTrain_fold = zeros(1448,10);
        mistakeTest_fold = zeros(4,1);
        for fold = 1:4
            for epoch = 1:5
                for t = 1:1448
                    %calculate the predict Y in training dataset
                    Yt_fold_predict(t,:) = transpose(alpha_fold{fold})*K_fold{fold}(:,t);
                    signTrain_fold(t,:) = sign(Yt_fold_predict(t,:)-0.000000001);

                    %update the alpha value
                    for n = 1:10
                        %update the alpha value if values are not the same
                        if Y_fold{fold}(t,n) * Yt_fold_predict(t,n) <=0
                             alpha_fold{fold}(t,n) = alpha_fold{fold}(t,n) - signTrain_fold(t,n);
                        end
                    end

                   %predict the value of y
                   [m,n] = max(Yt_fold_predict(t,:));
                   Yvalue_fold(t) = n-1;
                end
            end

            mistakeTrainMatrix = Yvalue_fold - Y_fold_real{fold};
            mistakeTrain = nnz(mistakeTrainMatrix);

            Yt_predict_test_fold = transpose(transpose(alpha_fold{fold})*transpose((X_fold{5}* transpose(X_fold{fold})).^d));
            %predict the value of Y_test
            [~,Yvalue_test_fold] = max(Yt_predict_test_fold,[],2);
            mistakeTestMatrix = Yvalue_test_fold - Y_fold_real{5} - 1;
            mistakeTest_fold(fold) = nnz(mistakeTestMatrix);
        end     
        mean_fold(d) = mean(mistakeTest_fold);
    end

    %find the position of the best d
    [~,dimension(run)] = min(mean_fold);
    K = (Xtrain(:,:)*Xtrain(:,:)').^dimension(run);
    alpha = zeros(7440,10);
    
    %calculate alpha row by row
    for Epoch = 1:5
        for t = 1:7440
            %calculate the predict Y in training dataset
            Yt_predict(t,:) = transpose(alpha(:,:))*K(:,t);
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
    end       
 %% testing
    % Calculate the predict value in test dataset
    Yt_predict_test(:,:) = transpose(transpose(alpha(:,:))*transpose((Xtest(:,:)* transpose(Xtrain(:,:))).^dimension(run)));

    %Calculate raw totals of mistake test
    [m_test,Yvalue_test] = max(Yt_predict_test,[],2);
    
    %Change the mistake in test to percentages
    meanTest(run)= nnz(Yvalue_test - Ytest - 1)/1858;
end

%Calculate mean test error,std and d
MeanDim = mean(dimension);
StdDim = std(dimension);
MeanTestError = mean(meanTest);
StdTestError = std(meanTest);

