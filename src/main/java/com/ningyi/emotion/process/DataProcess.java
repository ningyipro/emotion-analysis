package com.ningyi.emotion.process;

import com.ningyi.emotion.helper.HanLPUtils;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Component;

import java.io.*;
import java.util.*;

@Slf4j
@Component
/*
  数据处理
  TODO: 关注文件输出地址，如 Windows 自行修改对应地址
  TODO: 程序运行通过单测执行
 */
public class DataProcess {

    /**
     * 情绪标签
     */
    private final Map<String, Integer> emotionLabelMap = new HashMap<>();


    /**
     * 训练模型并分析语句
     *
     * @param args 一或多条语句
     * @throws IOException IOExp
     */
    public void buildModelAndAnalysisSentences(String... args) throws IOException {

        // 语句与情绪
        List<String> sentences = new ArrayList<>();
        List<String> emotions = new ArrayList<>();

        // 读取训练数据
        ClassPathResource classPathResource = new ClassPathResource("train/emotion.txt");
        InputStream inputStream = null;
        try {
            inputStream = classPathResource.getInputStream();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        InputStreamReader isr = new InputStreamReader(inputStream);
        BufferedReader br = new BufferedReader(isr);
        String line;
        while ((line = br.readLine()) != null) {
            String[] content = line.split(",");
            sentences.add(content[0]);
            emotions.add(content[1]);
        }
        br.close();

        // hanLP 分词
        List<List<String>> tokenizedSentences = new ArrayList<>();
        for (String sentence : sentences) {
            List<String> tokens = HanLPUtils.segment(sentence);
            tokenizedSentences.add(tokens);
        }

        // 将情绪标签转换为数值标签
        emotionLabelMap.put("敷衍", 0);
        emotionLabelMap.put("正常", 1);
        emotionLabelMap.put("开心", 2);
        emotionLabelMap.put("愤怒", 3);
        emotionLabelMap.put("我也不知道啦", 3);

        long[] emotionLabelsArray = emotions.stream()
                .map(emotionLabelMap::get)
                .mapToLong(Integer::longValue)
                .toArray();

        // 将情绪标签转换为独热编码
        int numClasses = 5;
        int batchSize = emotions.size();
        INDArray labels = Nd4j.create(Nd4j.createBuffer(emotionLabelsArray), new long[]{batchSize, numClasses});

        /* 构建情感分析模型 */

        // 特征大小
        Set<String> uniqueTokens = new HashSet<>();
        for (List<String> tokens : tokenizedSentences) {
            uniqueTokens.addAll(tokens);
        }

        // 嵌入大小
        int embeddingSize = 100;

        // 多层网络模型
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(0.01))
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(uniqueTokens.size() * embeddingSize)
                        .nOut(128)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(128)
                        .nOut(numClasses)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(configuration);

        // 模型建立
        model.init();

        // 数据处理
        DataSetIterator dataSetIterator = getDataIterator(tokenizedSentences, labels, uniqueTokens.size(), emotions.size(), embeddingSize);

        // 模型训练 100 次
        int numEpochs = 100;
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            model.fit(dataSetIterator);
        }

        // 模型评估
        Evaluation evaluation = model.evaluate(dataSetIterator);
        System.out.println("***** Evaluation *****");
        System.out.println(evaluation.stats());

        // 生成模型
        File modelFile = new File("/tmp/model.zip");
        model.save(modelFile, true);

        List<String> newSentences = Arrays.asList(args);

        MultiLayerNetwork loadedModel = null;
        try {
            loadedModel = ModelSerializer.restoreMultiLayerNetwork(modelFile);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        // 准备输入数据
        List<List<String>> tokenizedNewSentences = newSentences.stream()
                .map(HanLPUtils::segment)
                .toList();

        // 输入数据转换为特征向量
        List<INDArray> newFeatures = tokenizedNewSentences.stream()
                .map(tokens -> sentenceToFeatures(tokens, uniqueTokens.size(), 100))
                .toList();

        // 进行预测
        for (INDArray newFeature : newFeatures) {
            INDArray output = loadedModel.output(newFeature, false);
            System.out.println("模型预测结果：" + output);
            String emotion = predictEmotionLabel(output);
            for (String arg : args) {
                System.out.println("她说的是:[ " + arg + " ]" + ",当前她的情绪是:[ " + emotion + " ]");
            }
        }
    }

    /**
     * 情绪标签
     *
     * @param output 神经网络输出
     * @return String 情绪
     */
    public String predictEmotionLabel(INDArray output) {
        int predictedIndex = Nd4j.argMax(output, 1).getInt(0);
        return getEmotionLabelByIndex(predictedIndex);
    }

    /**
     * 情绪标签
     *
     * @param index 高概率索引
     * @return String 情绪
     */
    private String getEmotionLabelByIndex(int index) {
        for (Map.Entry<String, Integer> entry : emotionLabelMap.entrySet()) {
            if (entry.getValue() == index) {
                return entry.getKey();
            }
        }
        return "模型数据不足,无法评估";
    }

    /**
     * 训练模型
     *
     * @param tokenizedSentences 分词
     * @param labels             标签
     * @param uniqueTokensSize   嵌入维度
     * @param emotions           情感
     * @param embeddingSize      嵌入维度
     * @return DataSetIterator
     */
    private static DataSetIterator getDataIterator(List<List<String>> tokenizedSentences, INDArray labels, Integer uniqueTokensSize, Integer emotions, Integer embeddingSize) {
        List<DataSet> dataSets = new ArrayList<>();
        for (int i = 0; i < tokenizedSentences.size(); i++) {
            List<String> tokens = tokenizedSentences.get(i);
            INDArray features = sentenceToFeatures(tokens, uniqueTokensSize, embeddingSize);
            INDArray label = labels.getRow(i, true);
            dataSets.add(new DataSet(features, label));
        }
        return new ListDataSetIterator<>(dataSets, emotions);
    }

    /**
     * 将语句转换为特征向量  one-hot encoding
     *
     * @param tokens           句子单词
     * @param uniqueTokensSize 嵌入维度
     * @param embeddingSize    嵌入维度
     * @return INDArray 特征向量
     */
    private static INDArray sentenceToFeatures(List<String> tokens, int uniqueTokensSize, int embeddingSize) {
        INDArray features = Nd4j.create(1, uniqueTokensSize * embeddingSize);
        for (String token : tokens) {
            int index = hashFunction(token) % (uniqueTokensSize * embeddingSize);
            features.putScalar(index, 1.0);
        }
        return features;
    }

    /**
     * Hash 码
     *
     * @param token 句子单词
     * @return int hashcode
     */
    private static int hashFunction(String token) {
        //简单用 hash 处理
        return token.hashCode();
    }


}
