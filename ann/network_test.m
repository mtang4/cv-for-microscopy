ann1=load('ann1.mat');
net=ann1.net;

time=300;
t=1:time;
params=100:100:10000;

testParams=100:500:10000;
testSize = length(testParams);
testSet = zeros(time,testSize);

testError1 = zeros(6,testSize);

% known params
count=1;
for a=1:testSize
    for b=1:testSize
        for c=1:testSize
            i1 = params(a);
            i2 = params(b);
            i3 = params(c);

            tau1 = 5*rand(1);
            tau2 = 5*rand(1);
            tau3 = 5*rand(1);
        
            alpha = [i1 i2 i3 tau1 tau2 tau3];
            counts = i1 .* exp(-t*tau1) + i2 .* exp(-t*tau2) + i3 .* exp(-t*tau3);
            output = net(counts');
            
            testError1(:,count) = abs(output - alpha');
        end
    end
end
fprintf("total KNOWN parameter test error: \n")
mean(testError1,2)

% random params
testError2 = zeros(6,testSize);

for d=1:testSize
    i1 = randi([100 10000],1,1);
    i2 = randi([100 10000],1,1);
    i3 = randi([100 10000],1,1);
    tau1 = 5*rand(1);
    tau2 = 5*rand(1);
    tau3 = 5*rand(1);
    alpha = [i1 i2 i3 tau1 tau2 tau3];

    counts = i1 .* exp(-t*tau1) + i2 .* exp(-t*tau2) + i3 .* exp(-t*tau3);
    output = net(counts');
    testError2(:,d) = abs(output - alpha');
end

fprintf("total RANDOM parameter test error: \n")
mean(testError2,2)