import { useState, useEffect } from 'react'
import {
  Container,
  Header,
  SpaceBetween,
  Button,
  Table,
  Box,
  TextFilter
} from '@cloudscape-design/components'
import { apiService } from '../services/api'

interface DataItem {
  id: string
  name: string
  status: string
  created_at: string
}

export default function Dashboard() {
  const [items, setItems] = useState<DataItem[]>([])
  const [loading, setLoading] = useState(false)
  const [filteringText, setFilteringText] = useState('')

  const loadData = async () => {
    setLoading(true)
    try {
      const data = await apiService.getItems()
      setItems(data)
    } catch (error) {
      console.error('Failed to load data:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
  }, [])

  const filteredItems = items.filter(item =>
    item.name.toLowerCase().includes(filteringText.toLowerCase())
  )

  return (
    <SpaceBetween size="l">
      <Header
        variant="h1"
        actions={
          <Button variant="primary" onClick={loadData}>
            Refresh
          </Button>
        }
      >
        Dashboard
      </Header>

      <Container>
        <Table
          columnDefinitions={[
            {
              id: 'name',
              header: 'Name',
              cell: (item: DataItem) => item.name
            },
            {
              id: 'status',
              header: 'Status',
              cell: (item: DataItem) => item.status
            },
            {
              id: 'created_at',
              header: 'Created',
              cell: (item: DataItem) => new Date(item.created_at).toLocaleDateString()
            }
          ]}
          items={filteredItems}
          loading={loading}
          loadingText="Loading items..."
          empty={
            <Box textAlign="center" color="inherit">
              <b>No items</b>
              <Box padding={{ bottom: 's' }} variant="p" color="inherit">
                No items to display.
              </Box>
            </Box>
          }
          filter={
            <TextFilter
              filteringText={filteringText}
              onChange={({ detail }: { detail: { filteringText: string } }) => setFilteringText(detail.filteringText)}
              filteringPlaceholder="Find items"
            />
          }
          header={<Header>Items</Header>}
        />
      </Container>
    </SpaceBetween>
  )
}