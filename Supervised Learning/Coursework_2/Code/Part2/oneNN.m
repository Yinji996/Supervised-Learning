
clear;
max_n=15;
test_size=1500;
test_times=10;

parfor count=1:test_times
    disp(count)
    for N=1:max_n
        M=1;
        while(1)
            [X_train,Y_train]=random_dataset(M,N);
            [X_test,Y_test]=random_dataset(test_size,N); %generate the test dataset
            
            difference=zeros(1,M);
            error=0;
            for count_test=1:test_size
                for count_trained=1:M
                    difference(1,count_trained)=sum((X_test(count_test,:)-X_train(count_trained,:)).^2,2);
                end
                position=find(difference==min(difference));
                if Y_test(count_test,1)~=Y_train(position(1),1)
                    error=error+1;
                end
            end
            
            gerror=error/test_size;
            if gerror <0.1
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
errorbar(1:max_n,desired, stdand_derivation);
title('1-nearest neighbours');
ylabel('m');
xlabel('n');