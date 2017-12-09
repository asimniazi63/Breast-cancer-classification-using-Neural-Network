%load overall data download from repositry
data = load('breastCancer.m');

%taking mean of column 7 to replace 0 (which is misplaced data) with mean
%value

%var = mean(data(:,7)) %replaced 0 with 3

trainingData = data(:,2:10); %EXCLUDING patient ID
classData = data(:,11); %data on which NN will be classified

%prepare classifying data
classData(find(classData == 2)) = 0;
classData(find(classData == 4)) = 1;

%preparing traingdata 70% as training
trainInput = trainingData(1:490,1:9);
trainTarget = classData(1:490,1);

%for further experiments
%trainInput = trainingData(70:699,1:9);
%trainTarget = classData(70:699,1);

%preparing testingData 30% as testing
testInput = trainingData(490:699,1:9);
testTarget = classData(490:699,1);

%for further experiments
%testInput = trainingData(1:70,1:9);
%testTarget = classData(1:70,1);

%setting parameters
%net.LW{2,1};
%net.b{2};

net = newff(trainInput',trainTarget',2, {'tansig' 'tansig'}, 'traingdx', 'learngd', 'mse');
net.trainParam.epochs = 1000;
net.trainParam.lr  = 0.01;
net.trainParam.max_fail = 1000;
net.trainParam.goal = 0.01;

%training
net = train(net, trainInput', trainTarget');

%testing
testOutput = round(net(testInput'));
error = gsubtract(testTarget',testOutput); 
check = find(error ==0);
accuracy = length(check)/length(error) * 100;
disp(accuracy);







