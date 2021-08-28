time = 300;
i = 1:time;
h = 6;
t = 1:time;
t2 = i .* h;  % not sure -- histogram x-axis?

index=1;
params=100:50:10000;
len=length(params);
trainSize = len^3;
fprintf("trainset size: %f\n", trainSize)

trainSet = zeros(time,trainSize);
outputs = zeros(6,trainSize);


fprintf("Building training set:\n")

for i=1:len
    for j=1:len
        for k=1:len

            i1 = params(i);
            i2 = params(j);
            i3 = params(k);

            tau1 = 5*rand(1);
            tau2 = 5*rand(1);
            tau3 = 5*rand(1);

            % choose parameter values
            alpha = [i1 i2 i3 tau1 tau2 tau3];
            outputs(:,index)=alpha';

            % generate exponential decay curve
            counts = i1 .* exp(-t*tau1) + i2 .* exp(-t*tau2) + i3 .* exp(-t*tau3);
            noise = poissrnd(counts);
            trainSet(:,index) = noise';
            index = index + 1
        end
    end
end

net = feedforwardnet([10,50,10]);
net.trainParam.epochs = 50;
net.trainParam.max_fail = 20;
[net,tr] = train(net,trainSet,outputs);

% testing ------------------------------------------------------------------------
testParams=100:500:10000;
testSize = length(testParams);
testSet = zeros(time,testSize);

testError1 = zeros(1,testSize);

% known params
count=1;
for a=1:testSize
    for b=1:testSize
        for c=1:testSize
            i1 = params(i);
            i2 = params(j);
            i3 = params(k);

            tau1 = 5*rand(1);
            tau2 = 5*rand(1);
            tau3 = 5*rand(1);

            alpha = [i1 i2 i3 tau1 tau2 tau3];
            counts = i1 .* exp(-t*tau1) + i2 .* exp(-t*tau2) + i3 .* exp(-t*tau3);
            output = net(counts');

            testError1(count) = mean(abs(output - alpha'));
        end
    end
end

fprintf("total KNOWN test error: %f\n", mean(testError1))

% random params
testError2 = zeros(1,testSize);

for c=1:testSize
    i1 = randi([100 10000],1,1);
    i2 = randi([100 10000],1,1);
    i3 = randi([100 10000],1,1);
    tau1 = 5*rand(1);
    tau2 = 5*rand(1);
    tau3 = 5*rand(1);
    alpha = [i1 i2 i3 tau1 tau2 tau3];

    counts = i1 .* exp(-t*tau1) + i2 .* exp(-t*tau2) + i3 .* exp(-t*tau3);
    output = net(counts');
    testError2(c) = mean(abs(output - alpha'));
end

fprintf("total RANDOM test error: %f\n", mean(testError2))

