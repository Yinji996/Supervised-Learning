%winnow classification algorithm
clear;
max_n=100;
test_size=5000;
test_times=20;
for count=1:test_times
    for N=1:max_n
        M=1;
        
        while (1)
            gerror=0;
            %test_time=20;
            
            X_train = randi([0 1],M,N);  % creat a m*n random matrix from {0,1}
            Y_train = X_train(:,1);
            
%             maxEpoch= 1;
%             learningRate=1;
%             
%             [m,n]=size(x);
%             %x=[x ones(m,1)];
%             %x=[x ones(m,1)]; %adding the bias
            w=ones(N,1);
            %w=zeros(n+1,1);
            
%             for epoch=1:maxEpoch
%                 finish=true;
                for samlendex=1:M
                    if X_train(samlendex,:)*w>=N
                        judge=1;
                    elseif X_train(samlendex,:)*w <N
                        judge=0;
                    end
                    if judge~=Y_train(samlendex)
                        for index_n=1:N
                            w(index_n,1)=w(index_n,1)*2^((Y_train(samlendex)-judge)*X_train(samlendex,index_n));
                        end
                    end
                end
%                 if finish==true
%                     break;
%                 end
%             end
            
            %testing section
            X_test = randi([0 1],test_size,N);  % creat a m*n random matrix from {0,1}
            Y_test = X_test(:,1);
            error=0;
            for count_m=1:size(X_test,1)
                if X_test(count_m,:)*w >=N
                    judge1=1;
                elseif X_test(count_m,:)*w <N
                    judge1=0;
                end
                
                if judge1 ~=Y_test(count_m,1)
                    error=error+1;
                end
            end
%             generalisation_error_simple=error/M_test;
%             gerror=gerror+generalisation_error_simple;
            gerror=error/test_size;
%             gerror=gerror/test_time;
            if gerror<0.1
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
title('Winnow');
ylabel('m');
xlabel('n');