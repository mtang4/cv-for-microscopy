time = 300;
i = 1:time;
h = 6;
t = 1:time;
t2 = i .* h;  % not sure -- histogram x-axis?

index=1;
params=100:100:10000;
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
            index = index + 1;
        end
    end
end
fprintf("Completed training set.\n")
net = feedforwardnet([10,10]);
net.trainParam.epochs = 50;
net.trainParam.max_fail = 20;
fprintf("Begin training.\n")
[net,tr] = train(net,trainSet,outputs);

fprintf("Training completed.\n")
save net

% testing ------------------------------------------------------------------------
testParams=100:500:10000;
testSize = length(testParams);
testSet = zeros(time,testSize);

testError1 = zeros(6,testSize);

diary on  % write subsequent output to txt file
fprintf("step size = 100, smaller network (2 hidden layers, 10 nodes each).\n")
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
            count=count+1;
        end
    end
end

fprintf("total KNOWN test error:\n")
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

fprintf("total RANDOM test error:\n")
mean(testError2,2)
diary off
