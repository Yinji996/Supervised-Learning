% Least Square Algorithm
%close;
clear;
max_n=100;
test_size=5000;
test_times=25;

for count=1:test_times
for N=1:max_n
    M=1;
    %jud =1;
    while(1) %jud ==1
        %gerror=0;
        [X_train,Y_train]=random_dataset(M,N); 
        %
        
%         maxEpoch= 1;
%         learningRate=1;
        
%         [m,n]=size(x);
        %x=[x ones(m,1)];
        %x=[x ones(m,1)]; %adding the bias
        %w=zeros(n,1);
        %w=zeros(n+1,1);
        
        %         for epoch=1:maxEpoch
        %             finish=true;
        
        w=pinv(X_train)*Y_train;
        
        %             error=0;
        %             for samlendex=1:m
        %                 if x(samlendex,:)*w>=0
        %                     judge=1;
        %                 elseif x(samlendex,:)*w <0
        %                     judge=-1;
        %                 end
        %
        %                 %if sign(x(samlendex,:)*w)~=y(samlendex)
        %                 if judge~=y(samlendex)
        %                     error= error+1;
        %                     %finish=false;
        %                     %w=w+(learningRate*x(samlendex,:)*y(samlendex))';
        %                 end
        %             end
%         predict=zeros(m,1);
%         for t=1:size(predict,1)
%             predict= x*w;
%             if x(t,:)*w>=0
%                 predict(t,1)=1;
%             elseif x(t,:)*w <0
%                 predict(t,1)=-1;
%             end
%         end
        %predict =sign(x*w);
        
        %error= sum(y~=predict)/M;
        %             if finish==true
        %                 break;
        %             end
        %         end
        
        %expected_y=sign(x*w)-y;
%         M_test=500;
%         x_test=unidrnd(2,M_test,N);
%         for p=1:size(x_test,1)
%             for q=1:size(x_test,2)
%                 if x_test(p,q)==2
%                     x_test(p,q)= x_test(p,q)-3;
%                 else
%                     x_test(p,q)=1;
%                 end
%             end
%         end
%         
        [X_test,Y_test]=random_dataset(test_size,N); 
        
        %             for p=1:size(x_test,1)
        %                 for q=1:size(x_test,2)
        %                     if x_test(p,q)==2
        %                         x_test(p,q)= x_test(p,q)-3;
        %                     else
        %                         x_test(p,q)=1;
        %                     end
        %                 end
        %             end
        %y_test=x_test(:,1);
        error=0;
                for count_m=1:test_size
                    if X_test(count_m,:)*w >=0
                        judge1=1;
                    elseif X_test(count_m,:)*w <0
                        judge1=-1;
                    end
        
                    if judge1 ~=Y_test(count_m,1)
                        error=error+1;
                    end
                end
%         generalisation_error=error/M_test;
%         gerror=gerror+generalisation_error;
%         
%         gerror=gerror/test_time;
            gerror=error/test_size;
        if gerror<0.1
            break;%jud=0;
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
title('Least Squares');
ylabel('m');
xlabel('n');
% figure;
% plot(1:20,MM,'-');
%hold on;

% x_test_standard=zeros(2^N,N);
% %for count_
% error1=0;
% for count_m=1:m
%
%
%         if x_test(count_m,:)*w >=0
%             judge1=1;
%         elseif x_test(count_m,:)*w <0
%             judge1=-1;
%         end
%
%         if judge1 ~=y_test(count_m,1)
%             error=error+1;
%         end
%     %end
% end
% generalisation_error=2^(-m*n)*error;




