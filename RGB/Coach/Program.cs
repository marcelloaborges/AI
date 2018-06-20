using System;
using Network;

namespace Coach
{
    class Program
    {
        static void Main(string[] args)
        {
            var trainingArray = TrainingSet.Split("\n");

            var training = new float[trainingArray.Length - 1][];
            var desired = new float[trainingArray.Length - 1][];

            for (int i = 0; i < training.Length; i++)
            {
                var subSetTraining = trainingArray[i + 1].Replace("\r", "").Split(";");
                var color = subSetTraining[subSetTraining.Length - 1];

                training[i] = new float[subSetTraining.Length - 1];

                for (int j = 0; j < subSetTraining.Length - 1; j++)
                {
                    training[i][j] = float.Parse(subSetTraining[j]);
                }

                desired[i] = ColorToHotVector(color);
            }

            var standardizedTraining = Maths.Standardizer(training);

            var testing = new float[5][];
            testing[0] = new float[] {240, 245, 240}; //WHITE
            testing[1] = new float[] {75, 150, 10}; //GREEN 
            testing[2] = new float[] {228, 155, 112}; //RED
            testing[3] = new float[] {60, 95, 210}; //BLUE
            testing[4] = new float[] {240, 240, 55}; //YELLOW

            var standardizedTesting = Maths.Standardizer(testing);

            var testingLabeled = new string[5][];
            testingLabeled[0] = new[]
            {
                standardizedTesting[0][0] + "", standardizedTesting[0][1] + "", standardizedTesting[0][2] + "", "WHITE"
            };
            testingLabeled[1] = new[]
            {
                standardizedTesting[1][0] + "", standardizedTesting[1][1] + "", standardizedTesting[1][2] + "", "GREEN"
            };
            testingLabeled[2] = new[]
                {standardizedTesting[2][0] + "", standardizedTesting[2][1] + "", standardizedTesting[2][2] + "", "RED"};
            testingLabeled[3] = new[]
            {
                standardizedTesting[3][0] + "", standardizedTesting[3][1] + "", standardizedTesting[3][2] + "", "BLUE"
            };
            testingLabeled[4] = new[]
            {
                standardizedTesting[4][0] + "", standardizedTesting[4][1] + "", standardizedTesting[4][2] + "", "YELLOW"
            };

            
            
            int inputSize = 3;
            int hiddenSize = 5;
            int outputSize = 6;
            float weightsInitizalizationFactor = .1f;
            float learningRate = .3f;

            var network = new MLPSigmoidSoftmax(
                inputSize,
                hiddenSize,
                outputSize,
                weightsInitizalizationFactor,
                learningRate);

            network.Run(standardizedTraining, desired, testingLabeled, 1000000);
        }

        private static float[] ColorToHotVector(string color)
        {
            var enumColor = Enum.Parse<Color>(color);

            return ColorToHotVector(enumColor);
        }

        private static float[] ColorToHotVector(Color color)
        {
            var hotVec = new float[Enum.GetValues(typeof(Color)).Length];

            hotVec[(int) color] = 1;

            return hotVec;
        }

        private static string TrainingSet =>
            @"R;G;B;COLOR
255;255;255;WHITE
255;250;250;WHITE
240;255;240;WHITE
245;255;250;WHITE
240;255;255;WHITE
240;248;255;WHITE
248;248;255;WHITE
245;245;245;WHITE
255;245;238;WHITE
245;245;220;WHITE
253;245;230;WHITE
255;250;240;WHITE
255;255;240;WHITE
250;235;215;WHITE
250;240;230;WHITE
255;240;245;WHITE
255;228;225;WHITE
255;222;173;WHITE
0;0;0;BLACK
105;105;105;BLACK
128;128;128;BLACK
169;169;169;BLACK
192;192;192;BLACK
255;160;122;RED
250;128;114;RED
233;150;122;RED
240;128;128;RED
205;92;92;RED
220;20;60;RED
178;34;34;RED
255;0;0;RED
139;0;0;RED
128;0;0;RED
255;99;71;RED
255;69;0;RED
219;112;147;RED
124;252;0;GREEN
127;255;0;GREEN
50;205;50;GREEN
0;255;0;GREEN
34;139;34;GREEN
0;128;0;GREEN
0;100;0;GREEN
173;255;47;GREEN
154;205;50;GREEN
0;255;127;GREEN
0;250;154;GREEN
144;238;144;GREEN
152;251;152;GREEN
143;188;143;GREEN
60;179;113;GREEN
32;178;170;GREEN
46;139;87;GREEN
128;128;0;GREEN
85;107;47;GREEN
107;142;35;GREEN
240;248;255;BLUE
230;230;250;BLUE
176;224;230;BLUE
173;216;230;BLUE
135;206;250;BLUE
135;206;235;BLUE
0;191;255;BLUE
176;196;222;BLUE
30;144;255;BLUE
100;149;237;BLUE
70;130;180;BLUE
95;158;160;BLUE
123;104;238;BLUE
106;90;205;BLUE
72;61;139;BLUE
65;105;225;BLUE
0;0;255;BLUE
0;0;205;BLUE
0;0;139;BLUE
0;0;128;BLUE
25;25;112;BLUE
138;43;226;BLUE
75;0;130;BLUE
255;255;224;YELLOW
255;250;205;YELLOW
250;250;210;YELLOW
255;239;213;YELLOW
255;228;181;YELLOW
255;218;185;YELLOW
238;232;170;YELLOW
240;230;140;YELLOW
189;183;107;YELLOW
255;255;0;YELLOW
128;128;0;YELLOW
173;255;47;YELLOW
154;205;50;YELLOW";
    }
}