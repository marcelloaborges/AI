using FileHelpers;

namespace BestAd
{
    [DelimitedRecord(";"), IgnoreFirst(1)]
    public class Result
    {
        public string Id { get; set; }
        public int YesNo { get; set; }
    }
}