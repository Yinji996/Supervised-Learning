clear all;
%import data file
Data = dlmread('zipcombo.dat');

%obtain the size of the dataset
[Row,Column] = size(Data);

mistakeTrain = zeros(7,20);
mistakeTest = zeros(7,20);
%used to store the predicted value of Y
Yvalue = zeros(7440,1);
YvalueTest = zeros(1858,1);

%for calculate the expected Y
Y_09=meshgrid(0:9,1:7440);

predicted_train = zeros(7440,45); 
predicted_test = zeros(1858,45);  
%create one to one matrixs to calculate the predict value
matrix = [1:9;10:17,0;18:24,0,0;25:30,0,0,0;31:35,0,0,0,0;36:39,0,0,0,0,0;...
    40:42,0,0,0,0,0,0;43,44,0,0,0,0,0,0,0;45,0,0,0,0,0,0,0,0];

%store mean values in d=1:7 and 20 runs
dimension = zeros(1,20);
meanTest = zeros(1,20);
for run = 1:20
    A=randperm(Row);
    %use the fisrt 7440 rows in A as the trainning set
    Xtrain = Data (A(1:7440),2:257);
    Ytrain = Data (A(1:7440),1);
    %use the next 1858 rows in A as the testing set
    Xtest = Data (A(7441:9298),2:257);
    Ytest = Data (A(7441:9298),1);    
    
    
    %build 5 cell for 5-fold cross-validation
    X_fold = cell(5,1);Y_fold = cell(5,1);Y_fold_real = cell(5,1);
    for i = 1 : 5
        X_fold{i} = Xtrain(1448* (i-1) + 1 : 1448 * i,:);
        Y_fold{i} = Ytrain(1448* (i-1) + 1 : 1448 * i,:);
        Y_fold_real{i} = Ytrain(1448* (i-1) + 1 : 1448 * i);
    end
    
     %assign values to alpha
    alpha_fold = cell(4,1);K_fold = cell(4,1);
    Yvalue_fold = zeros(1448,1);
    for d= 1:7
        for i = 1:4
                alpha_fold{i} = zeros(1448,10);
                K_fold{i} = (X_fold{i}*X_fold{i}').^d;
        end
    end
        kTrain = transpose(Xtrain*(Xtrain)').^dimension(run);
    kTest = transpose(Xtest*(Xtrain)').^dimension(run);
    for fold = 1:4
        alpha = zeros(7440,45);
        for d = 1:7
           for Epoch = 1:2
                %create a K value to decide which one vs one predicted should taken
                Y = zeros(1858,45);
                %the compare begin with 0 vs 1, so the first value of k should be one
                Y(:,1) = 1;
                for t = 1:1858
                    for i = 1:45
                        %use the one to one matrix to find the two number which is compared
                        [row,column] = find(matrix == i);
                        small = row-1;
                        large = row+column-1;
                        %calculate the final predticed training value 
                        if Y(t,i) == 1 
                            predicted_train(t,i) = sign(sum(alpha(1:t,i).*kTrain(1:t,t))-0.000000001); 
                            %if the large number is 9,value can be obtained
                            if large == 9
                                if predicted_train(t,i)<0
                                Yvalue(t) = large;
                                else
                                Yvalue(t) = small;
                                end
                            else
                                if predicted_train(t,i)>0
                                    Y(t,i+1) = 1;
                                else
                                    Y(t,matrix(large+1,1)) = 1;
                                end
                            end
                        end
                        %update alpha
                        if Ytrain(t) == small
                            predicted_train(t,i) = sum(alpha(1:t,i).*kTrain(1:t,t));
                            predicted_train(t,i) = sign(predicted_train(t,i)-0.000000001);
                            if predicted_train(t,i)<0
                               alpha(t,i) = alpha(t,i) + 1;
                            end
                        elseif Ytrain(t) == large
                            predicted_train(t,i) = sum(alpha(1:t,i).*kTrain(1:t,t));
                            predicted_train(t,i) = sign(predicted_train(t,i)-0.000000001);
                            if predicted_train(t,i)>0
                               alpha(t,i) = alpha(t,i) - 1;
                            end
                        end
                     end 
                end
           end
        end
  
        mistakeTrain(d,run) = nnz(Ytrain-Yvalue);

        %% testing
        Y = zeros(1858,45);
        Y(:,1) = 1;
        for t = 1:1858
            for i = 1:45
                if Y(t,i) == 1 
                    predicted_test(t,i) = sign(sum(alpha(:,i).*kTest(:,t))-0.000000001);
                    [row,column] = find(matrix == i);
                    large = row+column-1;
                    small = row-1;
                    %if the large number is 9,value can be obtained
                    if large == 9
                       if predicted_test(t,i)<0
                           YvalueTest(t) = large;
                       else
                           YvalueTest(t) = small;
                       end
                       break;
                    else
                        if predicted_test(t,i)>0
                            Y(t,i+1) = 1;
                        else
                            Y(t,matrix(large+1,1)) = 1;
                        end
                    end
                end
            end
        end
    end
   
    %assign values to alpha
    alpha = zeros(7440,45);
    %Calculate kernel
    kTrain = transpose(Xtrain*(Xtrain)').^dimension(run);
    kTest = transpose(Xtest*(Xtrain)').^dimension(run);

    %% Train the data
    for Epoch = 1:2
        %create a K value to decide which one vs one predicted should taken
        Y = zeros(7440,45);
        %the compare begin with 0 vs 1, so the first value of k should be one
        Y(:,1) = 1;
        for t = 1:7440
            for i = 1:45
                %use the one to one matrix to find the two number which is compared
                [row,column] = find(matrix == i);
                small = row-1;
                large = row+column-1;
                %calculate the final predticed training value 
                if Y(t,i) == 1 
                    predicted_train(t,i) = sign(sum(alpha(1:t,i).*kTrain(1:t,t))-0.000000001); 
                    %if the large number is 9,value can be obtained
                    if large == 9
                        if predicted_train(t,i)<0
                        Yvalue(t) = large;
                        else
                        Yvalue(t) = small;
                        end
                    else
                        if predicted_train(t,i)>0
                            Y(t,i+1) = 1;
                        else
                            Y(t,matrix(large+1,1)) = 1;
                        end
                    end
                end
                %update alpha
                if Ytrain(t) == small
                    predicted_train(t,i) = sum(alpha(1:t,i).*kTrain(1:t,t));
                    predicted_train(t,i) = sign(predicted_train(t,i)-0.000000001);
                    if predicted_train(t,i)<0
                       alpha(t,i) = alpha(t,i) + 1;
                    end
                elseif Ytrain(t) == large
                    predicted_train(t,i) = sum(alpha(1:t,i).*kTrain(1:t,t));
                    predicted_train(t,i) = sign(predicted_train(t,i)-0.000000001);
                    if predicted_train(t,i)>0
                       alpha(t,i) = alpha(t,i) - 1;
                    end
                end
             end 
        end
    end
    mistakeTrain(d,run) = nnz(Ytrain-Yvalue);

    %% testing
    Y = zeros(1858,45);
    Y(:,1) = 1;
    for t = 1:1858
        for i = 1:45
            if Y(t,i) == 1 
                predicted_test(t,i) = sign(sum(alpha(:,i).*kTest(:,t))-0.000000001);
                [row,column] = find(matrix == i);
                large = row+column-1;
                small = row-1;
                %if the large number is 9,value can be obtained
                if large == 9
                   if predicted_test(t,i)<0
                       YvalueTest(t) = large;
                   else
                       YvalueTest(t) = small;
                   end
                   break;
                else
                    if predicted_test(t,i)>0
                        Y(t,i+1) = 1;
                    else
                        Y(t,matrix(large+1,1)) = 1;
                    end
                end
            end
        end
    end
    %calculate the test error and test error rate
    mistakeTest(run) = nnz(Ytest - YvalueTest);
end

%% Calculate mean test error,std and d
meanTest = mistakeTest / 1858;

MeanDim = mean(dimension);
StdDim = std(dimension);
MeanTestError = mean(meanTest);
StdTestError = std(meanTest);