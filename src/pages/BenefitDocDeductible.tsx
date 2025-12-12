import { useState } from 'react'
import {
  Container,
  Header,
  SpaceBetween,
  Button,
  Box,
  Alert,
  Table,
  Badge,
  ColumnLayout,
  KeyValuePairs
} from '@cloudscape-design/components'

interface DeductibleInfo {
  id: string
  documentName: string
  status: 'completed' | 'processing' | 'failed'
  deductibleData: {
    individualDeductible: string
    familyDeductible: string
    inNetworkDeductible: string
    outOfNetworkDeductible: string
    deductibleType: string
    planYear: string
  }
  extractedAt: string
}

export default function BenefitDocDeductible() {
  const [loading, setLoading] = useState(false)
  const [deductibles, setDeductibles] = useState<DeductibleInfo[]>([
    {
      id: '1',
      documentName: 'Health_Benefits_2024.pdf',
      status: 'completed',
      deductibleData: {
        individualDeductible: '$1,500',
        familyDeductible: '$3,000',
        inNetworkDeductible: '$1,500',
        outOfNetworkDeductible: '$3,500',
        deductibleType: 'Calendar Year',
        planYear: '2024'
      },
      extractedAt: '2024-12-10T10:30:00Z'
    },
    {
      id: '2',
      documentName: 'PPO_Plan_Details.pdf',
      status: 'completed',
      deductibleData: {
        individualDeductible: '$2,000',
        familyDeductible: '$4,000',
        inNetworkDeductible: '$2,000',
        outOfNetworkDeductible: '$5,000',
        deductibleType: 'Plan Year',
        planYear: '2024'
      },
      extractedAt: '2024-12-09T14:20:00Z'
    }
  ])

  const handleExtractDeductibles = async () => {
    setLoading(true)
    // Simulate API call
    setTimeout(() => {
      setLoading(false)
    }, 1500)
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'completed':
        return <Badge color="green">Completed</Badge>
      case 'processing':
        return <Badge color="blue">Processing</Badge>
      case 'failed':
        return <Badge color="red">Failed</Badge>
      default:
        return <Badge>Unknown</Badge>
    }
  }

  return (
    <SpaceBetween size="l">
      <Header
        variant="h1"
        description="Extract deductible information from benefit documents"
        actions={
          <Button
            variant="primary"
            onClick={handleExtractDeductibles}
            loading={loading}
          >
            Extract Deductibles
          </Button>
        }
      >
        Benefit Document - Deductible Extraction
      </Header>

      <Alert type="info">
        This specialized extraction focuses on deductible amounts, including individual, family, in-network, and out-of-network deductibles from benefit documents.
      </Alert>

      <Container>
        <Table
          columnDefinitions={[
            {
              id: 'document',
              header: 'Document',
              cell: (item: DeductibleInfo) => (
                <SpaceBetween direction="horizontal" size="xs">
                  <Box>{item.documentName}</Box>
                  {getStatusBadge(item.status)}
                </SpaceBetween>
              )
            },
            {
              id: 'individual',
              header: 'Individual Deductible',
              cell: (item: DeductibleInfo) => item.deductibleData.individualDeductible
            },
            {
              id: 'family',
              header: 'Family Deductible',
              cell: (item: DeductibleInfo) => item.deductibleData.familyDeductible
            },
            {
              id: 'in-network',
              header: 'In-Network',
              cell: (item: DeductibleInfo) => item.deductibleData.inNetworkDeductible
            },
            {
              id: 'out-network',
              header: 'Out-of-Network',
              cell: (item: DeductibleInfo) => item.deductibleData.outOfNetworkDeductible
            },
            {
              id: 'type',
              header: 'Deductible Type',
              cell: (item: DeductibleInfo) => item.deductibleData.deductibleType
            }
          ]}
          items={deductibles}
          loading={loading}
          loadingText="Extracting deductible information..."
          empty={
            <Box textAlign="center" color="inherit">
              <b>No deductible data available</b>
              <Box padding={{ bottom: 's' }} variant="p" color="inherit">
                Upload benefit documents to extract deductible information.
              </Box>
            </Box>
          }
          header={<Header>Deductible Extraction Results</Header>}
        />
      </Container>

      {deductibles.length > 0 && (
        <Container>
          <Header variant="h2">Deductible Summary</Header>
          <ColumnLayout columns={2}>
            <KeyValuePairs
              columns={1}
              items={[
                {
                  label: 'Total Documents Processed',
                  value: deductibles.length.toString()
                },
                {
                  label: 'Average Individual Deductible',
                  value: '$1,750'
                },
                {
                  label: 'Average Family Deductible',
                  value: '$3,500'
                }
              ]}
            />
            <KeyValuePairs
              columns={1}
              items={[
                {
                  label: 'Lowest Individual Deductible',
                  value: '$1,500'
                },
                {
                  label: 'Highest Individual Deductible',
                  value: '$2,000'
                },
                {
                  label: 'Most Common Deductible Type',
                  value: 'Calendar Year'
                }
              ]}
            />
          </ColumnLayout>
        </Container>
      )}
    </SpaceBetween>
  )
}