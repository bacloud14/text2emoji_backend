import com.google.common.base.Splitter;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
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
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.*;

import java.util.*;
import java.util.logging.Logger;

public class MAIN {
    private final static Logger LOGGER = Logger.getLogger(MAIN.class.getName());
    private final static int ngrams = 3;
    private final static int topNearest = 3;

    public static ArrayList<String> readStopWords(String path) throws FileNotFoundException {
        Scanner s = new Scanner(new File(path));
        ArrayList<String> list = new ArrayList<String>();
        while (s.hasNext()) {
            list.add(s.next());
        }
        s.close();

        return list;
    }

    public static void main(String[] args) throws IOException {
        String stopWordsPath = new ClassPathResource("stopwords.txt").getFile().getAbsolutePath();
        ArrayList<String> stopWords = readStopWords(stopWordsPath);
        String filePath = new ClassPathResource("xaa").getFile().getAbsolutePath();
        String emojisPath = new ClassPathResource("emojis.json").getFile().getAbsolutePath();

//        String glovePath = new ClassPathResource("glove.6B.100d.txt").getFile().getAbsolutePath();
        LOGGER.info("Load & Vectorize Sentences....");
        SentenceIterator iter = new BasicLineIterator(filePath);

        // Split on white spaces in the line to get words
        TokenizerFactory biTokenizer = new NGramTokenizerFactory(new DefaultTokenizerFactory(), 1, ngrams); //new DefaultTokenizerFactory();
        biTokenizer.setTokenPreProcessor(new CommonPreprocessor());


        Runtime.getRuntime().gc();

        JSONParser parser = new JSONParser();
        ArrayList<String[]> rows = new ArrayList<String[]>();
        try {
            JSONArray jsonArray = (JSONArray) parser.parse(new FileReader(emojisPath));

            for (int i = 0; i < jsonArray.size(); i++) {
                JSONObject jsonobject = (JSONObject) jsonArray.get(i);
                String[] row = new String[4];
                row[0] = (String) jsonobject.get("name");
                row[1] = (String) jsonobject.get("unicode");
                String category = (String) jsonobject.get("category");
                if (category == null) category = "other";
                JSONArray keywords = (JSONArray) jsonobject.get("keywords");
                String row2 = "";
                for (int j = 0; j < keywords.size(); j++) {
                    row2 = row2 + " " + keywords.get(j);
                }
                row[2] = row2;
                row[3] = category;

                rows.add(row);
            }
        } catch (ParseException e) {
            e.printStackTrace();
        }
        Runtime.getRuntime().gc();
        ArrayList<Collection<String>> emojisVectors = new ArrayList();
        Word2Vec vec = new Word2Vec.Builder()
                .stopWords(stopWords)
                .minWordFrequency(5)
                .iterations(3)
                .layerSize(100)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(biTokenizer)
                .build();

        Runtime.getRuntime().gc();
        LOGGER.info("Fitting Word2Vec model....");
        vec.fit();

        WordVectorSerializer.writeWord2VecModel(vec, "model");

//        Word2Vec vec = WordVectorSerializer.readWord2VecModel(new File("model"));


        int idxBackSpace = 0;
        for (String[] r : rows) {
            idxBackSpace++;
            if (idxBackSpace % 100 == 50) {
                System.out.println(".");
                System.out.println(r[2]);
            } else
                System.out.print(".");
            // pick keywords or names
            Collection<String> lst_2 = vec.wordsNearest(r[2].trim(), 3);
            if (lst_2.isEmpty()) {
                String name = r[0].toLowerCase().replaceAll("[^A-Za-z]+", "");
                lst_2 = vec.wordsNearest(name, 2);
            }
            emojisVectors.add(lst_2);
        }
        System.out.println(".");
        System.out.println(emojisVectors);
//
//
//        try {
//            FileOutputStream fos = new FileOutputStream("emojisVectors");
//            ObjectOutputStream oos = new ObjectOutputStream(fos);
//            oos.writeObject(emojisVectors);
//            oos.close();
//            fos.close();
//        } catch (IOException ioe) {
//            ioe.printStackTrace();
//        }


//        try {
//            FileInputStream fis = new FileInputStream("emojisVectors");
//            ObjectInputStream ois = new ObjectInputStream(fis);
//
//            emojisVectors = (ArrayList<Collection<String>>) ois.readObject();
//
//            ois.close();
//            fis.close();
//        } catch (IOException ioe) {
//            ioe.printStackTrace();
//            return;
//        } catch (ClassNotFoundException c) {
//            System.out.println("Class not found");
//            c.printStackTrace();
//            return;
//        }
        check(vec, emojisVectors, rows, "star");
        check(vec, emojisVectors, rows, "nice people");
        check(vec, emojisVectors, rows, "black man");
        check(vec, emojisVectors, rows, "very excited");
        check(vec, emojisVectors, rows, "skateboard snow");

    }

    private static void check(Word2Vec vec, ArrayList<Collection<String>> emojisVectors, ArrayList<String[]> rows, String sentence) {
        System.out.println("\n\nSentence: " + sentence);
        String sentence_ = String.join(" ", vec.wordsNearest(sentence.trim(), 1));
        if (sentence_.equals(""))
            sentence_ = sentence;
        double max = 0;
        int idx = -1;
        int det = -1;
        Collection<String> bestEmojiVector = emojisVectors.get(0);
        for (Collection<String> emojiVector : emojisVectors) {
            idx++;
            if (emojiVector.isEmpty())
                continue;
            double score = cosineSimForSentence(vec, String.join(" ", emojiVector), sentence_);
            if (score > max) {
                max = score;
                bestEmojiVector = emojiVector;
                det = idx;
            }

        }
        System.out.println("det " + det);
        System.out.println("bestEmojiVector " + bestEmojiVector.toString());
        System.out.print("row " + Arrays.toString(rows.get(det)));
        System.out.print("max " + max);

    }

    public static double cosineSimForSentence(Word2Vec vector, String sentence1, String sentence2) {
        Collection<String> label1 = Splitter.on(' ').splitToList(sentence1);
        Collection<String> label2 = Splitter.on(' ').splitToList(sentence2);
        try {
            return Transforms.cosineSim(vector.getWordVectorsMean(label1), vector.getWordVectorsMean(label2));
        } catch (Exception e) {
            String exceptionMessage = e.getMessage();
            System.out.print(exceptionMessage);
        }
        return Transforms.cosineSim(vector.getWordVectorsMean(label1), vector.getWordVectorsMean(label2));

    }

}
