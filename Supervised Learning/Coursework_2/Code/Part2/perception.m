%
clear;
max_n=100;
test_size=5000;
test_times=20;

for count=1:test_times
    for N=1:max_n
        M=1;
        while(1)
            
            %trainning section 
            [X_train,Y_train]=random_dataset(M,N); %generate the trainning dataset
            w=zeros(N,1); %initialize the weight
            
            %update weight
            for index=1:M
                if X_train(index,:)*w*Y_train(index)<=0
                    w=w+Y_train(index)*X_train(index,:)';
                end
            end
            
            %testing section
            [X_test,Y_test]=random_dataset(test_size,N); %generate the test dataset
            
            prediction=sign(X_test*w);
            prediction(prediction==0)=1;
            error=sum(prediction~=Y_test)/test_size  ;
            
            %             generalisation_error_simple=error/M_test;
            %             gerror=gerror+generalisation_error_simple;
            
            %gerror=gerror/test_time;
            if error<0.1
                break;
            else
                M=M+1;
            end
        end
        M_set(N,count)=M;    
    end
end

desired=sum(M_set,2)/test_times;
stdand_derivation=std(M_set,0,2);
errorbar(1:max_n,desired , stdand_derivation);
title('Perceptron');
ylabel('m');
xlabel('n');






