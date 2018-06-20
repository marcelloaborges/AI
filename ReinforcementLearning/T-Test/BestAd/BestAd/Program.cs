using System;
using System.Linq;
using FileHelpers;

namespace BestAd
{
    class Program
    {
        static void Main(string[] args)
        {
            var engine = new FileHelperEngine<Result>();

            var url = "C:\\Dev\\Learning\\Base\\Ad\\advertisement_clicks.csv";

            var results = engine.ReadFile(url);


            /////////////////////////////////////////
            var aResults = results.Where(x => x.Id == "A").ToList();
            var aN = aResults.Count;

            var bResults = results.Where(x => x.Id == "B").ToList();
            var bN = bResults.Count;

            //H0 => BOTH ADs HAVE THE SAME RESULT (THE NULL HYPOTHESES)
            //H1 => ADs HAVE A DIFFERENT RESULT WITH HIGH SIGNIFICANT RELEVANCE (ALTERNATIVE HYPOTHESES)
                        
            //CALCULATE TEST STATISTIC

            //MEAN
            var aMean = results.Where(x => x.Id == "A" && Math.Abs(x.YesNo - 1d) <= 0).Sum(x => x.YesNo) / (double) aN;
            var bMean = results.Where(x => x.Id == "B" && Math.Abs(x.YesNo - 1d) <= 0).Sum(x => x.YesNo) / (double) bN;

            //VARIANCE^2
            var aux = 0d;

            foreach (var asr in aResults)
            {
                aux += Math.Pow(asr.YesNo - aMean, 2);
            }

            var aVariance = aux / (aN - 1);

            aux = 0d;
            foreach (var bsr in bResults)
            {
                aux += Math.Pow(bsr.YesNo - bMean, 2);
            }

            var bVariance = aux / (bN - 1);

            //T-VALUE
            var t = (aMean - bMean) / Math.Sqrt(aVariance / aN + bVariance / bN);
            
            //ALPHA => THE % OF ERROR RATE
            //USALLY FOR SCIENCE, ANYTHING BIGGER THAN 5% IS BAD
            var alpha = 0.05d;

            //STATE DECISION RULE => t TABLE
            //FOR 0.95, WITH 5% OF SIGNIFICANT ERROR, THE t VALUE IS => -2.0167 to 2.0167 FOR 30 AS DF
            //OUT OF THIS RANGE, REJECT H0 (-2.0167 <= T <= 2.0167)
            
            //DEGREES OF FREEDOM
            var df = (aN - 1) + (bN - 1);
        }
    }
}