using System;
using FileHelpers;
using System.Linq;

namespace CTRTest
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


            var clicksA = 36d;//aResults.Count(x => x.YesNo == 1);    //A
            var noClickA = 14d;//aResults.Count(x => x.YesNo == 0);  //B

            var clicksB = 30d;//bResults.Count(x => x.YesNo == 1);    //C
            var noClickB = 25d;//bResults.Count(x => x.YesNo == 0);  //D

            var x21 = Math.Pow((clicksA * noClickB) - (noClickA * clicksB), 2) *
                     (clicksA + noClickA + clicksB + noClickB);

            var x22 = (clicksA + noClickA) * (clicksB + noClickB) * (clicksA + clicksB) * (noClickA + noClickB);
 
            var x2 = x21 / x22;
        }
    }
}