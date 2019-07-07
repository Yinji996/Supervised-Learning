clear all;
%import data file
Data = dlmread('zipcombo.dat');

%define parameters
epoch = 20;

%obtain the size of the dataset
[Row,Column] = size(Data);

%used to store the predicted value of Y
Yvalue = zeros(9298,1);
%Yt_predict:predict whether the value is larger than 0
Yt_predict = zeros(9298,10);
signTrain = zeros(9298,10);

%for calculate the expected Y
Y_09=meshgrid(0:9,1:9298);

A=randperm(Row);
%use the whole rows in A as the trainning set
Xtrain = Data (A(:),2:257);
Ytrain = Data (A(:),1);

%find the expected value
Y(:,:) = Y_09(:,:) - Ytrain(:).*ones(1,10);
Y(Y~=0)=-1;
Y(Y==0)=1;

K = (Xtrain(:,:)*Xtrain(:,:)').^3;
transK = K';
alpha = zeros(9298,10);

%% train the data
for Epoch = 1:epoch
    for t = 1:9298
        %calculate the predict Y in training dataset
        Yt_predict(t,:) = transpose(alpha(:,:))*transK(:,t);
        signTrain(t,:) = sign(Yt_predict(t,:)-0.000000001);

        %update the alpha value
        for n = 1:10
            %calculate the predict Y in training dataset
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
    if mistakeTrain < 6
        break;
    end
end       
%% show image
position = find(mistakeTrainMatrix);
for i = 1:size(position,1)
    figure;
    image = reshape(Xtrain(position(i),:),16,16)';
    image = image/2+0.5;

    I = imresize(image,10);
    imshow(I);
    title(Ytrain(position(i)));
end
