import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;

import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.logging.Logger;

public class MAIN {
    public static String text = "Marie was born in Paris.";
    private final static Logger LOGGER = Logger.getLogger(MAIN.class.getName());

    public static void main(String[] args) throws IOException {
        String filePath = new ClassPathResource("raw_sentences.txt").getFile().getAbsolutePath();
        String emojisPath = new ClassPathResource("emojis.json").getFile().getAbsolutePath();

        String glovePath = new ClassPathResource("glove.6B.100d.txt").getFile().getAbsolutePath();
        LOGGER.info("Load & Vectorize Sentences....");
        SentenceIterator iter = new BasicLineIterator(filePath);

        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        WordVectors wordVectors = WordVectorSerializer.loadTxtVectors(new File(glovePath));
//        WordVectorSerializer.writeWordVectors(wordVectors.lookupTable(), "wow.model");
        Collection<String> lst_ = wordVectors.wordsNearest("day", 10);
        System.out.println(Arrays.toString(lst_.toArray()));


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
                    row2 = row2 + " "+ keywords.get(j);
                }
                row[2] = row2;
                rows.add(row);
            }
        } catch (ParseException e) {
            e.printStackTrace();
        }
        for (String[] r:rows
             ) {
            System.out.println(r[2]);
            Collection<String> lst_2 = wordVectors.wordsNearest(r[2], 10);
            System.out.println(Arrays.toString(lst_2.toArray()));
            break;
        }

//        Word2Vec vec = new Word2Vec.Builder()
//                .minWordFrequency(5)
//                .layerSize(100)
//                .windowSize(5)
//                .iterate(iter)
//                .tokenizerFactory(t)
//                .build();
//
//        LOGGER.info("Fitting Word2Vec model....");
//        vec.fit();
//
//        WordVectorSerializer.writeWordVectors(vec, "pathToWriteto.txt");
//
//        LOGGER.info("Closest Words:");
//        Collection<String> lst = vec.wordsNearest("day", 8);
//        System.out.println(lst);
    }

}
