package com.ningyi.emotion.helper;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.seg.common.Term;

import java.util.ArrayList;
import java.util.List;

/**
 * hanPl
 */
public class HanLPUtils {

    /**
     * 分词
     *
     * @param sentence 语句
     * @return 分词结果
     */
    public static List<String> segment(String sentence) {
        List<Term> termList = HanLP.segment(sentence);
        List<String> tokens = new ArrayList<>();
        for (Term term : termList) {
            tokens.add(term.word);
        }
        return tokens;
    }
}
