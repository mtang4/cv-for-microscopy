trainSize = 1000;
time = 256;  
i = 1:time;
h = 6;
t = 1:300;
t2 = i .* h;  % not sure -- histogram x-axis?

trainSet = zeros(300,trainSize);
outputs = zeros(3,trainSize);

for i=1:trainSize
    % choose parameter values
    K=randi(50,1); 
    % f_D=rand(1); 
    i1 = randi(100,1);
    i2 = randi(100,1);
    tau_F = 5*rand(1); 
    tau_D = 5*rand(1);
    alpha = [i1 i2 tau_F tau_D];
    outputs(:,i)=alpha;
    
    % generate exponential decay curve
    counts = i1 .* exp(-t*tau_F) + i2 .* exp(-t*tau_D);
    noise = poissrnd(counts);
    trainSet(:,i) = noise' + counts';
    
    % compute MLE target outputs
    expected = K * (tau_F*f_D * exp((-i .* h)/tau_F) * (exp(h/tau_F) - 1) ...
        + tau_D*(1-f_D) * exp((-i .* h)/tau_D) * (exp(h/tau_D) - 1));
    
end

net = feedforwardnet([10,20]);
net.trainParam.epochs = 50;
net.trainParam.max_fail = 20;
[net,tr] = train(net,trainSet,outputs);

testSize = 50;
testSet = zeros(300,testSize);

testError = zeros(1,50);

for j=1:testSize
    K=randi(50,1);
    f_D=rand(1); 
    tau_F = 5*rand(1); 
    tau_D = 5*rand(1);
    alpha = [f_D tau_F tau_D];
    
    counts = f_D .* exp(-t*tau_F) + (1-f_D) .* exp(-t*tau_D);
    output = net(counts');
    testError(j) = mean(abs(output - alpha'));
end

mean(testError)
