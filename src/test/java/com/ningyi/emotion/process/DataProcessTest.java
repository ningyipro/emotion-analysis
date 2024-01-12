package com.ningyi.emotion.process;

import jakarta.annotation.Resource;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
class DataProcessTest {

    @Resource
    private DataProcess dataProcess;

    @Test
    void buildModelAndAnalysisSentences() {
        try {
            dataProcess.buildModelAndAnalysisSentences("你吃了吗");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}