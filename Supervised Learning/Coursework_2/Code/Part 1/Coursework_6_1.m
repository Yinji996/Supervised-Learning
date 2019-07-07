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

predicted_train = zeros(7440,45); 
predicted_test = zeros(1858,45);  
%create one to one matrixs to calculate the predict value
matrix = [1:9;10:17,0;18:24,0,0;25:30,0,0,0;31:35,0,0,0,0;36:39,0,0,0,0,0;...
    40:42,0,0,0,0,0,0;43,44,0,0,0,0,0,0,0;45,0,0,0,0,0,0,0,0];

for run = 1:20
    A=randperm(Row);
    %use the fisrt 7440 rows in A as the trainning set
    Xtrain = Data (A(1:7440),2:257);
    Ytrain = Data (A(1:7440),1);
    %use the next 1858 rows in A as the testing set
    Xtest = Data (A(7441:9298),2:257);
    Ytest = Data (A(7441:9298),1);    

    for d = 1:7
        %assign values to alpha
        alpha = zeros(7440,45);
        %Calculate kernel
        kTrain = transpose(Xtrain*(Xtrain)').^d;
        kTest = transpose(Xtest*(Xtrain)').^d;

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
                        predicted_train(t,i) = sign(alpha(1:t,i).*kTrain(1:t,t)-0.000000001);
                        if predicted_train(t,i)<0
                           alpha(t,i) = alpha(t,i) + 1;
                        end
                    elseif Ytrain(t) == large
                        predicted_train(t,i) = sign(sum(alpha(1:t,i).*kTrain(1:t,t))-0.000000001);
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
        mistakeTest(d,run) = nnz(Ytest - YvalueTest);
    end
end

%% Calculate mean test error and standard deviation
meanTest = mistakeTest / 1858;
meanTrain = mistakeTrain / 7440;

result_Train = zeros(d,2);
result_Train(:,1)=mean(meanTrain,2);
result_Train(:,2)=std(meanTrain,0,2);

result_Test = zeros(d,2);
result_Test(:,1)=mean(meanTest,2);
result_Test(:,2)=std(meanTest,0,2);