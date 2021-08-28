trainSize = 100^3;
time = 300;  
i = 1:time;
h = 6;
t = 1:time;
t2 = i .* h;  % not sure -- histogram x-axis?

trainSet = zeros(time,trainSize);
outputs = zeros(6,trainSize);

index=1;
params=100:100:10000;
len=length(params);
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

net = feedforwardnet([10,50,10]);
net.trainParam.epochs = 50;
net.trainParam.max_fail = 20;
[net,tr] = train(net,trainSet,outputs);

testSize = 50;
testSet = zeros(time,testSize);

testError = zeros(1,50);

for j=1:testSize
    i1 = randi(100000,1);
    i2 = randi(100000,1);
    i3 = randi(100000,1);
    tau1 = 5*rand(1); 
    tau2 = 5*rand(1);
    tau3 = 5*rand(1);
    alpha = [i1 i2 i3 tau1 tau2 tau3];
    
    counts = i1 .* exp(-t*tau1) + i2 .* exp(-t*tau2) + i3 .* exp(-t*tau3);
    output = net(counts');
    testError(j) = mean(abs(output - alpha'));
end

mean(testError)
