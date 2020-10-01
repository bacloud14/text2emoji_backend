import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;

import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.NGramTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import org.nd4j.linalg.io.ClassPathResource;

import java.io.*;

import java.util.*;
import java.util.logging.Logger;

public class MAIN {
    public static String text = "Marie was born in Paris.";
    private final static Logger LOGGER = Logger.getLogger(MAIN.class.getName());

    public static void main(String[] args) throws IOException {
        String filePath = new ClassPathResource("xaa").getFile().getAbsolutePath();
        String emojisPath = new ClassPathResource("emojis.json").getFile().getAbsolutePath();

//        String glovePath = new ClassPathResource("glove.6B.100d.txt").getFile().getAbsolutePath();
        LOGGER.info("Load & Vectorize Sentences....");
        SentenceIterator iter = new BasicLineIterator(filePath);

        // Split on white spaces in the line to get words
        TokenizerFactory t = new NGramTokenizerFactory(new DefaultTokenizerFactory(), 1, 3); //new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());


        Runtime.getRuntime().gc();

        JSONParser parser = new JSONParser();
        ArrayList<String[]> rows = new ArrayList<String[]>();
        try {
            JSONArray jsonArray = (JSONArray) parser.parse(new FileReader(emojisPath));

            for (int i = 0; i < jsonArray.size(); i++) {
                JSONObject jsonobject = (JSONObject) jsonArray.get(i);
                String[] row = new String[3];
                row[0] = (String) jsonobject.get("name");
                row[1] = (String) jsonobject.get("unicode");
                JSONArray keywords = (JSONArray) jsonobject.get("keywords");
                String row2 = "";
                for (int j = 0; j < keywords.size(); j++) {
                    row2 = row2 + " " + keywords.get(j);
                }
                row[2] = row2;
                rows.add(row);
            }
        } catch (ParseException e) {
            e.printStackTrace();
        }
        ArrayList<Collection<String>> emojisVectors = new ArrayList();
//        Word2Vec vec = new Word2Vec.Builder()
//                .minWordFrequency(10)
//                .layerSize(64)
//                .windowSize(5)
//                .iterate(iter)
//                .tokenizerFactory(t)
//                .build();
//
//        Runtime.getRuntime().gc();
//        LOGGER.info("Fitting Word2Vec model....");
//        vec.fit();
//
//        WordVectorSerializer.writeWordVectors(vec, "pathToWriteto.txt");

        String cached_model = new ClassPathResource("pathToWriteto.txt").getFile().getAbsolutePath();
        Word2Vec vec = WordVectorSerializer.readWord2Vec(new File("pathToWriteto.txt"));

        //        WordVectorSerializer.readWord2Vec()
        LOGGER.info("Closest Words:");
        Collection<String> lst = vec.wordsNearest("happy world", 5);
        System.out.println(lst);

        int idxBackSpace = 0;
        for (String[] r : rows) {
            idxBackSpace++;
            if (idxBackSpace % 100 == 0) {
                System.out.println(".");
                System.out.println(r[2]);
            } else
                System.out.print(".");

            Collection<String> lst_2 = vec.wordsNearest(r[2].trim(), 1);
            emojisVectors.add(lst_2);
        }
        System.out.println(".");
        System.out.println(emojisVectors);
//        int idxBackSpace = 0;
//        for (String[] r : rows) {
//            idxBackSpace++;
//            if (idxBackSpace % 100 == 0)
//                System.out.println(".");
//            else
//                System.out.print(".");
//
//            Collection<String> lst_2 = wordVectors.wordsNearest(
////                    Arrays.asList("hello world".split(" ")),
//                    Arrays.asList(r[2].trim().split(" ")),
//                    Arrays.asList("the"),
//                    5
//            );
//            emojisVectors.add(lst_2);
//        }
//        try
//        {
//            FileOutputStream fos = new FileOutputStream("listData");
//            ObjectOutputStream oos = new ObjectOutputStream(fos);
//            oos.writeObject(emojisVectors);
//            oos.close();
//            fos.close();
//        }
//        catch (IOException ioe)
//        {
//            ioe.printStackTrace();
//        }


//        try
//        {
//            FileInputStream fis = new FileInputStream("listData");
//            ObjectInputStream ois = new ObjectInputStream(fis);
//
//            emojisVectors = (ArrayList<Collection<String>>) ois.readObject();
//
//            ois.close();
//            fis.close();
//        }
//        catch (IOException ioe)
//        {
//            ioe.printStackTrace();
//            return;
//        }
//        catch (ClassNotFoundException c)
//        {
//            System.out.println("Class not found");
//            c.printStackTrace();
//            return;
//        }
//
//
//
//        similarity("hello happy world peace together", emojisVectors, wordVectors);


    }

    private static void similarity(String sentence, ArrayList<Collection<String>> emojisVectors, WordVectors wordVectors) {

        Collection<String> lst = wordVectors.wordsNearest(
                Arrays.asList(sentence.split(" ")),
                Arrays.asList("the"),
                5
        );
        System.out.print(lst);
        double max = 0;
        Collection<String> bestEmojiVector = emojisVectors.get(0);
        for (Collection<String> emojiVector : emojisVectors) { // reference vectors to compare to // out one vector
            double innerSum = 0;
            for (String keyWord : emojiVector) { // one emoji vector to compare to
                for (String queryWord : lst) {
                    innerSum += wordVectors.similarity(keyWord, queryWord);
                }
                if (innerSum > max) {
                    max = innerSum;
                    bestEmojiVector = emojiVector;
                }
                innerSum = 0;
            }
        }
        System.out.println("max: " + max);
        System.out.print("bestEmojiVector: " + bestEmojiVector.toString());
    }


    public static double cosineSimilarity(double[] vectorA, double[] vectorB) {
        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;
        for (int i = 0; i < vectorA.length; i++) {
            dotProduct += vectorA[i] * vectorB[i];
            normA += Math.pow(vectorA[i], 2);
            normB += Math.pow(vectorB[i], 2);
        }
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }
}
